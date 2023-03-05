package main

import "fmt"

func program1(a, b float32) float32 {
	c := a + b
	return c
}

func main() {
	result := program1(23.4, 10.5)
	fmt.Println("c = a + b: ", result)
}
