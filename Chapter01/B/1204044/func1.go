package main

import "fmt"

func func1(a int, b int) int {
	c := a + b
	return c
}

func main() {
	a := 7
	b := 14
	c := func1(a, b)
	fmt.Printf("%d + %d = %d\n", a, b, c)
}
