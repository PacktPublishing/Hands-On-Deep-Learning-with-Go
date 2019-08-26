package main

import (
	"fmt"
	"strings"
)

const START rune = 0x02
const END rune = 0x03
const BLANK rune = 0x04

// vocab related
var sentences []string
var vocab []rune
var vocabIndex map[rune]int
var maxsent int = 30

func initVocab(ss []string, thresh int) {
	s := strings.Join(ss, " ")
	fmt.Println(s)
	dict := make(map[rune]int)
	for _, r := range s {
		dict[r]++
	}

	vocab = append(vocab, START)
	vocab = append(vocab, END)
	vocab = append(vocab, BLANK)
	vocabIndex = make(map[rune]int)

	for ch, c := range dict {
		if c >= thresh {
			// then add letter to vocab
			vocab = append(vocab, ch)
		}
	}

	for i, v := range vocab {
		vocabIndex[v] = i
	}
	// vocabIndex[START] = 0
	// vocabIndex[END] = 1
	// vocabIndex[BLANK] = 2

	fmt.Println("Vocab: ", vocab)
	inputSize = len(vocab)
	outputSize = len(vocab)
	epochSize = len(ss)
	fmt.Println("\ninputs :", inputSize)
	fmt.Println("\noutputs :", outputSize)
	fmt.Println("\nepochs: :", epochSize)
	fmt.Println("\nmaxsent: :", maxsent)
}

func init() {
	sentencesRaw := strings.Split(corpus, "\n")

	for _, s := range sentencesRaw {
		s2 := strings.TrimSpace(s)
		if s2 != "" {
			sentences = append(sentences, s2)
		}

	}

	initVocab(sentences, 1)
}
