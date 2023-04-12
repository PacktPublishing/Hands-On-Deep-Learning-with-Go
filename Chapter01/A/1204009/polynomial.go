package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func main() {
	g := gorgonia.NewGraph()

	// Define input tensor
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	z := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("z"))

	// Define output tensor using polynomial function
	y := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, x)), gorgonia.Must(gorgonia.Mul(x, z))))

	// Create a VM to run the graph
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	gorgonia.Let(x, 4.0)
	gorgonia.Let(z, 10.0)

	// Run the graph with input value 2.0
	if err := machine.RunAll(); err != nil {
		fmt.Println(err)
		return
	}

	// Print the output value
	fmt.Printf("\nHasil : %+v\n", y.Value())
}
