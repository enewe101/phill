package main

import (
	"os"
	"strings"
)


func main() {

	usage := "Usage:\n\"go run *.go <format> <mode> <inPathsFile> <outDir>\"\n" +
	"<format> should be \"conllu\" or \"plain\"\n" +
	"<mode> should be \"dict\" or \"index\"\n" +
	"<inPathsFile> should be the path to a file listing all input files to " +
	"be processed\n" +
	"<outDir> will be created by `dict` command if it doesn't exist.)"

	// If we have the right number of arguments..
	if len(os.Args) != 5 {
		panic(usage)
	} else {

		// Get the commandline arguments.
		format := os.Args[1]
		mode := os.Args[2]
		inPaths := ReadInputPathsFile(os.Args[3])
		outDir := os.Args[4]

		// Handle the plain file format.
		if format == "plain" {
			if mode == "dict" {
				makeDictionary(inPaths, outDir, plainDictParser)
			} else if mode == "index" {
				indexSentences(inPaths, outDir, plainSentenceParser)
			} else {
				panic("Unrecognized input file format: " + format + "\n" + usage)
			}

		// Handle the conllu file format.
		} else if format == "conllu" {
			if mode == "dict" {
				makeDictionary(inPaths, outDir, conlluDictParser)
			} else if mode == "index" {
				indexSentences(inPaths, outDir, conlluSentenceParser)
			} else {
				panic("Unrecognized input file format: " + format + "\n" + usage)
			}
		} else {
			panic("Unrecognized mode: " + mode + "\n" +usage)
		}
	}
}


func ReadInputPathsFile(path string) []string {

	// Read in the file that lists paths for all input files.
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err.Error())
	}

	// Collect the paths: one per line.  Ignore blank lines.
	paths := []string{}
	for _, path := range strings.Split(string(data), "\n") {
		path = strings.TrimSpace(path)
		if path != "" {
			paths = append(paths, path)
		}
	}
	return paths
}

