package main

import "fmt"

func main() {
    a := 10
    b := 20
    c := tambah(a, b)
    fmt.Println("Hasil penjumlahan", a, "dan", b, "adalah", c)
}

func tambah(a int, b int) int {
    return a + b
}
