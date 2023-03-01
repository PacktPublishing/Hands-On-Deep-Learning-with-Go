package main

import "fmt"

func program1(a, b float64) float64 {
    c := a + b
    return c
}

func main() {
    result := program1(2.5, 3.8)
    fmt.Println(result)
}
