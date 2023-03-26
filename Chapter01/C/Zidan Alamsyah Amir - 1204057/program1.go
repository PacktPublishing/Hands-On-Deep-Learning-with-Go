package main_1

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func main_1() {
	g := gorgonia.NewGraph()

	var a, b, c *gorgonia.Node
	var err error

	// define the expression
	a = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))
	if c, err = gorgonia.Add(a, b); err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// set initial values then run
	gorgonia.Let(a, 1.0)
	gorgonia.Let(b, 2.0)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v", c.Value())
}
