package main

import (
	"io"
	"os"
	"bytes"
	"sync"
)

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


func NewChunkerGroup(
	inPath string,
	chunkSize int64,
	peekSize int64,
	delim []byte,
	numChunkers int) []*Chunker {

	// Open the file.
	file, err := os.Open(inPath)
	if err != nil {panic(err.Error()+" ("+inPath+")")}

	// Make the chunkers.
	var chunkers []*Chunker
	numChunkers64 := int64(numChunkers)
	var waitGroup sync.WaitGroup
	waitGroup.Add(numChunkers)
	for i:=int64(0); i<numChunkers64; i++ {
		chunker := NewChunker(
			file, chunkSize, peekSize, delim,
			i, numChunkers64, &waitGroup)
		chunkers = append(chunkers, &chunker)
	}

	// Set a process to close the file once all chunkers are closed
	go closeFileWhenDone(file, &waitGroup)
	return chunkers
}

// Closes a file once all readers are done.
func closeFileWhenDone(file *os.File, waitGroup *sync.WaitGroup) {
	defer file.Close()
	waitGroup.Wait()
}


type Chunker struct {
	file *os.File
	chunkNum int64
	chunkSize int64
	peekSize int64
	delim []byte
	id int64
	numPeers int64
	waitGroup *sync.WaitGroup
}

func NewChunker(
	file *os.File,
	chunkSize int64,
	peekSize int64,
	delim []byte,
	id int64,
	numPeers int64,
	waitGroup *sync.WaitGroup) Chunker {
	return Chunker{
		chunkNum: -1,
		file: file,
		chunkSize: chunkSize,
		peekSize: peekSize,
		delim: delim,
		id: id,
		numPeers: numPeers,
		waitGroup: waitGroup}
}

func(c *Chunker) Close() {
	c.waitGroup.Done()
}

func(c *Chunker) next() ([]byte, error) {

	c.chunkNum++
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
	offset := (c.chunkNum * c.numPeers + c.id) * c.chunkSize 
	nominal_chunk, nominal_err := readAt(c.file, offset, c.chunkSize)
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
			c.chunkNum++
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
			c.chunkNum * c.numPeers + c.id + 1) * c.chunkSize +
			i * c.peekSize
		this_peek, peek_err := readAt(c.file, this_peek_offset, c.peekSize)
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

