package main

import (
	"io"
	"os"
	"bytes"
)

func NewChunkerGroup(
	file *os.File, chunk_size int64, peek_size int64, delim []byte,
		num_chunkers int) []*Chunker {
	var chunkers []*Chunker
	num_chunkers_ := int64(num_chunkers)
	for i:=int64(0); i<num_chunkers_; i++ {
		chunker := NewChunker(file, chunk_size, peek_size, delim)
		chunker.id = i
		chunker.num_peers = num_chunkers_
		chunkers = append(chunkers, &chunker)
	}
	return chunkers
}

func readAt(file *os.File, offset int64, length int64) ([]byte, error) {
	buf := make([]byte, length)
	len_read, err := file.ReadAt(buf, offset)
	buf = buf[:len_read]

	// file.ReadAt will return EOF even if len_read has some bytes.
	// but we only want to issue EOF when we couldn't get any
	if err == io.EOF && len_read > 0 {
		err = nil
	}
	return buf, err
}

type Chunker struct {
	file *os.File
	chunk_num int64
	chunk_size int64
	peek_size int64
	delim []byte
	id int64
	num_peers int64
}

func NewChunker(
	file *os.File, chunk_size int64, peek_size int64, delim []byte) Chunker {
	return Chunker{
		chunk_num: -1,
		file: file,
		chunk_size: chunk_size,
		peek_size: peek_size,
		delim: delim,
		num_peers: 1}
}

func(c *Chunker) next() ([]byte, error) {

	c.chunk_num++
	nominal_chunk, err := c.get_nominal_chunk()

	if err != nil {
		// If we reached EOF, we're done.
		if err == io.EOF {
			return nil, io.EOF
		} else {
			panic(err)
		}
	}

	// Add last sentence remainder
	trailing_sentence := c.get_trailing_sentence()
	chunk := append(nominal_chunk, trailing_sentence...)

	return chunk, nil
}

func (c *Chunker) get_nominal_chunk() ([]byte, error) {

	// Read up to the nominal size of the chunk
	offset := (c.chunk_num * c.num_peers + c.id) * c.chunk_size 
	nominal_chunk, nominal_err := readAt(c.file, offset, c.chunk_size)
	if nominal_err == io.EOF {
		return nominal_chunk, io.EOF
	}

	// Remove the first sentence, which belongs to last chunk (except if this is
	// the first chunk!)
	if offset != 0 {
		split := bytes.SplitN(nominal_chunk, c.delim, 2)

		// If the first token takes up the entire chunk, we need not process
		// this chunk at all, it's been done already.  Get the next one.
		if len(split) == 1 {
			c.chunk_num++
			return c.get_nominal_chunk()
		}
		nominal_chunk = split[len(split)-1]
	}
	return nominal_chunk, nil
}

func (c *Chunker) get_trailing_sentence() ([]byte) {

	// Read a bit more to finish off the last sentence in this chunk
	peek_chunk := []byte{}
	var i int64 = -1
	for {
		i++
		this_peek_offset := (
			c.chunk_num * c.num_peers + c.id + 1) * c.chunk_size +
			i * c.peek_size
		this_peek, peek_err := readAt(c.file, this_peek_offset, c.peek_size)
		peek_chunk = append(peek_chunk, this_peek...)

		// if we reach EOF, just return everything.
		if peek_err != nil {
			if peek_err == io.EOF {
				return peek_chunk
			}
			// Panic on unexpected error.
			panic(peek_err.Error())
		}

		// Try to find the first delimiter.  I.e. split should make two pieces.
		split := bytes.SplitN(peek_chunk, c.delim, 2)

		// if we found it, we're done
		if len(split) == 2 {
			peek_chunk = split[0]
			return peek_chunk
		}
		// otherwise, we peek further until we find it or reach EOF
	}
}
