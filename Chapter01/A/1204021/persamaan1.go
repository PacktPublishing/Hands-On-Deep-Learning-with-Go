package main

import (
	"fmt"
	"log"

	// "io/ioutil"
	G "gorgonia.org/gorgonia"
)

// kode program fungsi : c = a + b
func main() {
	g := G.NewGraph()

	var a, b, c *G.Node
	var err error

	// define the expression
	a = G.NewScalar(g, G.Float64, G.WithName("a"))
	b = G.NewScalar(g, G.Float64, G.WithName("b"))

	// fungsi c=a+b
	if c, err = G.Add(a, b); err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := G.NewTapeMachine(g)
	defer machine.Close()

	// set initial values then run
	G.Let(a, 3.0)
	G.Let(b, 3.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	// ioutil.WriteFile("pers1_graph.dot", []byte(g.ToDot()), 0644)

	fmt.Println(c.Value().Data())
}
