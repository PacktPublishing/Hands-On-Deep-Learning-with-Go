package main
import (
    "fmt"
    // "log"
    . "gorgonia.org/gorgonia"
)

func main() {
	g := NewGraph()	
	a := NewScalar(g, Float64, WithName("a"))
	b := NewScalar(g, Float64, WithName("b"))
	c, _ := Add(a,b)
	machine := NewTapeMachine(g)
	Let(a, 50.0)
	Let(b, 5.0)
	machine.RunAll()
	fmt.Printf("Nilai c = %s + %s", a.Value(), b.Value())
	fmt.Printf("\nNilai c = %s", c.Value())
}
