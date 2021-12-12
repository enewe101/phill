package main

import (
	"os"
	"fmt"
)

func main() {
	usage := "Usage:\n\"main dict inPath outDir\" " +
	"or \"main index inPath outDir\" " +
	"(outDir will be created by `dict` command if it doesn't exist.)"
	if len(os.Args) < 2 {
		fmt.Println(usage)
		os.Exit(1)
	}
	if os.Args[1] == "dict" && len(os.Args) == 4 {
		inPath := os.Args[2]
		outDir := os.Args[3]
		makeDictionary(inPath, outDir)
	} else if os.Args[1] == "index" && len(os.Args) == 4 {
		inPath := os.Args[2]
		outDir := os.Args[3]
		indexSentences(inPath, outDir)
	} else {
		fmt.Println(usage)
		os.Exit(1)
	}
}
