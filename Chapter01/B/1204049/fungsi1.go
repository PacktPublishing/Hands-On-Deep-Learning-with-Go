package main

import (
	"fmt"

	. "gorgonia.org/gorgonia"
)

func main() {
	g := NewGraph()
	a := NewScalar(g, Float64, WithName("a"))
	b := NewScalar(g, Float64, WithName("b"))
	c, _ := Add(a, b)

	machine := NewTapeMachine(g)
	Let(a, 230.0)
	Let(b, 7.0)
	machine.RunAll()
	fmt.Printf("Nilai a = %v\n", a.Value())
	fmt.Printf("Nilai b = %v\n", b.Value())
	fmt.Printf("Nilai c = a + b")
	fmt.Printf("\nNilai c = %v", c.Value())
}
