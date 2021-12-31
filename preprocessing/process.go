package main

import (
	"os"
	"io/fs"
	"errors"
	"fmt"
	"strings"
)

const sentenceFname = "sentences.index"
const tokenDictFname = "tokens.dict"
const relationDictFname = "relations.dict"
const Mb = int64(1024 * 1024)
const Kb = int64(1024)
const numWorkersPerFile = 4

// Parser functions that factor out file-format specific parsing.
type dictParser func(*[]byte) (*[]string, *[]string)
type sentenceParser func(*[]byte, *Dictionary, *Dictionary) (*[]string)


// Parse all of the files in `inPaths`, using `parser`.  Read dictionaries in
// outDir or create them if they don't exist, and add relations and tokens to
// them generated by the `parser`.  Write the dictionaries to disk at `outDir`.
func makeDictionary(inPaths []string, outDir string, parser dictParser) {

	// We do some non-newline-containing prints.  Clean up with a newline.
	defer fmt.Println()

	// Create (if noexist) the directory where output will be written.
	err := os.Mkdir(outDir, 0777)
	if err != nil && !errors.Is(err, fs.ErrExist) {panic(err.Error())}

	// Read dictionaries, if they exist, or create them.
	tokenDictPath := outDir + "/" + tokenDictFname
	tokenDict := readOrCreateDictionary(
		tokenDictPath, []string{"<ROOT>", "<UNK>", "<PAD>"})
	relationDictPath := outDir + "/" + relationDictFname
	relationDict := readOrCreateDictionary(relationDictPath, []string{"self"})

	// Parse each file, and add the parsed tokens to the dictionaries.
	for i, inPath := range inPaths {
		data, err := os.ReadFile(inPath)
		if err != nil {panic(err.Error())}
		newTokens, newRelations := parser(&data)
		tokenDict.Add(*newTokens...)
		relationDict.Add(*newRelations...)
		// Every once in awhile print the progress..
		printProgress(i, len(inPaths), 100)
	}

	// Write the dictionaries to disk.
	tokenDict.Write(tokenDictPath)
	relationDict.Write(relationDictPath)
}


// Parse all of the files in `inPaths`, using `parser`.  The parser creates
// strings representing sentences using ids assigned by the dictionaries.
// Write these strings to a file in outDir.  The dictionaries are those
// generated by makeDictionary, which should have been run on the same
// `inPaths` as this with the resulting .dict files located at `outDir`.
func indexSentences(inPaths []string, outDir string, parser sentenceParser) {

	// We do some non-newline-containing prints.  Clean up with a newline.
	defer fmt.Println()

	// Create file for sentence index.  outDir must exist.
	outfile, err := os.Create(outDir + "/" + sentenceFname)
	if err != nil {panic(err.Error())}
	defer outfile.Close()

	// Read dictionaries used to index sentences.
	tokenDictPath := outDir + "/" + tokenDictFname
	relationDictPath := outDir + "/" + relationDictFname
	tokenDict, err := ReadDictionary(tokenDictPath)
	if err != nil {panic(err.Error())}
	relationDict, err := ReadDictionary(relationDictPath)
	if err != nil {panic(err.Error())}

	// Parse each input file; write the parsed sentences to disk
	for i, inPath := range inPaths {
		data, err := os.ReadFile(inPath)
		if err != nil {panic(err.Error())}
		newSentences := parser(&data, tokenDict, relationDict)
		for _, sentence := range *newSentences {outfile.WriteString(sentence)}
		// Every once in awhile print the progress..
		printProgress(i, len(inPaths), 100)
	}
}


func readOrCreateDictionary(path string, initialTokens []string) *Dictionary {
	// Attempt to read a dictionary at path.  If none exists, make one, and
	// check that we can at least write to the dictionary path.
	var dictionary *Dictionary
	dictionary, err := ReadDictionary(path)
	if err != nil {
		// Create a new dictionary.  Check that the desired path is writable
		// to avoid a late IO error.
		checkWritable(path)
		dictionary = NewDictionary()
		dictionary.Add(initialTokens...)
	} else {
		fmt.Println("Using found dictionary.")
	}
	return dictionary
}


// Test writing to a given path.  Use this to discover common IO errors
// before expensive calculations generating results to be writen.
func checkWritable(path string) {
	file, err := os.Create(path)
	if err != nil {panic(err.Error())}
	file.Close()
}


// Every time `printEvery` parts are done, print `numPartsDone` as a
// percentage of `totalNumParts`.
func printProgress(numPartsDone int, totalNumParts int, printEvery int) {
	if numPartsDone % printEvery == 0 {
		percentComplete := float32(numPartsDone)/float32(totalNumParts)*100
		fmt.Fprintf(os.Stderr, strings.Repeat("\b", 30))
		fmt.Fprintf(os.Stderr, "files completed: %.2f%%", percentComplete)
	}
}

