package main

import (
	"fmt"
	"strings"
)

const START rune = 0x02
const END rune = 0x03

// vocab related
var sentences []string
var vocab []rune
var vocabIndex map[rune]int

func initVocab(ss []string, thresh int) {
	s := strings.Join(ss, " ")
	fmt.Println(s)
	dict := make(map[rune]int)
	for _, r := range s {
		dict[r]++
	}

	vocab = append(vocab, START)
	vocabIndex = make(map[rune]int)

	for ch, c := range dict {
		if c >= thresh {
			// then add letter to vocab
			vocab = append(vocab, ch)
		}
	}

	vocab = append(vocab, END)

	for i, v := range vocab {
		vocabIndex[v] = i
	}
	fmt.Println("Vocab: ", vocab)
	inputSize = len(vocab)
	outputSize = len(vocab)
	epochSize = len(ss)
	fmt.Println("\ninputs :", inputSize)
	fmt.Println("\noutputs :", outputSize)
	fmt.Println("\nepochs: :", epochSize)
}

func init() {
	sentencesRaw := strings.Split(corpus, "\n")
	// fmt.Printf("Type of raw: %T\n\n", sentencesRaw)
	for _, s := range sentencesRaw {
		s2 := strings.TrimSpace(s)
		if s2 != "" {
			sentences = append(sentences, s2)
		}
	}

	initVocab(sentences, 1)
}
