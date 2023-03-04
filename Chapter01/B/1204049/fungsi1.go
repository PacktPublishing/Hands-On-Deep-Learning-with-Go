package main

import "fmt"

func add(a, b int32) int32 {
	return a + b
}

func main() {
	c := add(23, 7)
	fmt.Println("a = 23")
	fmt.Println("b = 7")
	fmt.Println("c = a + b")
	fmt.Println("c =", c)
}
