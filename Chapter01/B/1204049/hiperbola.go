package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func main() {
	g := gorgonia.NewGraph()

	// Define variables
	a := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))

	// Calculate c. Jarak dari pusat ke titik asimtot: c = sqrt(a^2 + b^2)
	a2 := gorgonia.Must(gorgonia.Square(a))
	b2 := gorgonia.Must(gorgonia.Square(b))
	c := gorgonia.Must(gorgonia.Sqrt(gorgonia.Must(gorgonia.Add(a2, b2))))

	// Compile and run the graph
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	gorgonia.Let(a, 3.0)
	gorgonia.Let(b, 7.0)

	if err := machine.RunAll(); err != nil {
		fmt.Printf("Failed to run graph: %v\n", err)
		return
	}

	// Print the result
	cVal := c.Value()

	fmt.Println("Jarak dari pusat ke titik asimtot: c = sqrt(a^2 + b^2)")
	fmt.Printf("a = %v\n", a.Value())
	fmt.Printf("b = %v\n", b.Value())
	fmt.Printf("c = %v\n", cVal)
}
