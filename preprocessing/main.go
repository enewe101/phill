package main

import (
	"os"
	"fmt"
	"path"
)

var conlluPaths = []string{
	//"UD_English-Atis/en_atis-ud-train.conllu", // all "flight" stuff.
	//"UD_English-ESL/en_esl-ud-train.conllu", // no tokens, only dep relations
	"UD_English-EWT/en_ewt-ud-train.conllu", // Good.  A lot of news, some memos
	"UD_English-GUM/en_gum-ud-train.conllu", // academic, dialogue, disfluency
	//"UD_English-GUMReddit/en_gumreddit-ud-train.conllu", // only dep relations
	"UD_English-LinES/en_lines-ud-train.conllu", // technical, literature
	"UD_English-ParTUT/en_partut-ud-train.conllu"} // legal, literature

func main() {
	usage := "Usage:\n\"main dict inDir outDir\" " +
	"or \"main index inDir outDir\"\n" +
	"(inDir should be the path to the directory listing data for various " +
	"treebanks (e.g. \"path/to/ud-treebanks-v2.9\".))\n" +
	"(outDir will be created by `dict` command if it doesn't exist.)"

	// If we have the right number of arguments...
	if len(os.Args) < 3 {
		fmt.Println(usage)
		os.Exit(1)
	} else {

		// Set up full file paths.
		treebanksDir := os.Args[2]
		inPaths := make([]string, len(conlluPaths))
		for i, conlluPath := range conlluPaths {
			inPaths[i] = path.Join(treebanksDir, conlluPath)
		}
		outDir := os.Args[3]

		// Either make the dictionaries.
		if os.Args[1] == "dict" && len(os.Args) == 4 {
			makeDictionary(inPaths, outDir)

		// Or make the index.
		} else if os.Args[1] == "index" && len(os.Args) == 4 {
			indexSentences(inPaths, outDir)

		// Or do nothing.
		} else {
			fmt.Println(usage)
			os.Exit(1)
		}
	}
}
