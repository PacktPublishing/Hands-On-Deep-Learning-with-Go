package main

import "fmt"

func func3(W int, x int, b int) int {
	z := W*x + b
	return z
}

func main() {
	W := 7
	x := 14
	b := 2
	z := func3(W, x, b)
	fmt.Printf("%d*%d + %d = %d\n", W, x, b, z)
}
