package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func main() {
	// Create a new graph
	g := gorgonia.NewGraph()

	// Define input variables
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	a := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))
	r := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("r"))

	// Define symbolic expression for a circle
	x_minus_a := gorgonia.Must(gorgonia.Sub(x, a))
	x_minus_a_pow2 := gorgonia.Must(gorgonia.Mul(x_minus_a, x_minus_a))
	y_minus_b := gorgonia.Must(gorgonia.Sub(y, b))
	y_minus_b_pow2 := gorgonia.Must(gorgonia.Mul(y_minus_b, y_minus_b))
	r_pow2 := gorgonia.Must(gorgonia.Mul(r, r))
	circle := gorgonia.Must(gorgonia.Sub(gorgonia.Must(gorgonia.Add(x_minus_a_pow2, y_minus_b_pow2)), r_pow2))

	// Define output tensor using the circle expression
	output := circle

	// Create a VM to run the graph
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// Assign values to the input variables
	gorgonia.Let(x, 0.0)
	gorgonia.Let(y, 0.0)
	gorgonia.Let(a, 1.0)
	gorgonia.Let(b, 2.0)
	gorgonia.Let(r, 5.0)

	// Run the graph
	if err := machine.RunAll(); err != nil {
		fmt.Println(err)
		return
	}

	// Print the output value
	fmt.Printf("Circle equation: %v", output.Value())
}

//Hasilnya adalah
// Circle equation: -20
