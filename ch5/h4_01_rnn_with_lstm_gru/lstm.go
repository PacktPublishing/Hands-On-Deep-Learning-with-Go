package main

import (
	. "gorgonia.org/gorgonia"
)

type LSTM struct {
	wix    *Node
	wih    *Node
	bias_i *Node

	wfx    *Node
	wfh    *Node
	bias_f *Node

	wox    *Node
	woh    *Node
	bias_o *Node

	wcx    *Node
	wch    *Node
	bias_c *Node
}

func MakeLSTM(g *ExprGraph, hiddenSize, prevSize int) LSTM {
	retVal := LSTM{}

	retVal.wix = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wix_"))
	retVal.wih = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("wih_"))
	retVal.bias_i = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_i_"), WithInit(Zeroes()))

	// output gate weights

	retVal.wox = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wfx_"))
	retVal.woh = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("wfh_"))
	retVal.bias_o = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_f_"), WithInit(Zeroes()))

	// forget gate weights

	retVal.wfx = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wox_"))
	retVal.wfh = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("woh_"))
	retVal.bias_f = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_o_"), WithInit(Zeroes()))

	// cell write

	retVal.wcx = NewMatrix(g, Float, WithShape(hiddenSize, prevSize), WithInit(GlorotN(1.0)), WithName("wcx_"))
	retVal.wch = NewMatrix(g, Float, WithShape(hiddenSize, hiddenSize), WithInit(GlorotN(1.0)), WithName("wch_"))
	retVal.bias_c = NewVector(g, Float, WithShape(hiddenSize), WithName("bias_c_"), WithInit(Zeroes()))
	return retVal
}

func (l *LSTM) learnables() Nodes {
	return Nodes{
		l.wix, l.wih, l.bias_i,
		l.wfx, l.wfh, l.bias_f,
		l.wcx, l.wch, l.bias_c,
		l.wox, l.woh, l.bias_o,
	}
}

func (l *LSTM) Activate(inputVector *Node, prev lstmout) (out lstmout, err error) {
	// log.Printf("prev %v", prev.hidden.Shape())
	prevHidden := prev.hidden
	prevCell := prev.cell
	var h0, h1, inputGate *Node
	h0 = Must(Mul(l.wix, inputVector))
	h1 = Must(Mul(l.wih, prevHidden))
	inputGate = Must(Sigmoid(Must(Add(Must(Add(h0, h1)), l.bias_i))))

	var h2, h3, forgetGate *Node
	h2 = Must(Mul(l.wfx, inputVector))
	h3 = Must(Mul(l.wfh, prevHidden))
	forgetGate = Must(Sigmoid(Must(Add(Must(Add(h2, h3)), l.bias_f))))

	var h4, h5, outputGate *Node
	h4 = Must(Mul(l.wox, inputVector))
	h5 = Must(Mul(l.woh, prevHidden))
	outputGate = Must(Sigmoid(Must(Add(Must(Add(h4, h5)), l.bias_o))))

	var h6, h7, cellWrite *Node
	h6 = Must(Mul(l.wcx, inputVector))
	h7 = Must(Mul(l.wch, prevHidden))
	cellWrite = Must(Tanh(Must(Add(Must(Add(h6, h7)), l.bias_c))))

	// cell activations
	var retain, write *Node
	retain = Must(HadamardProd(forgetGate, prevCell))
	write = Must(HadamardProd(inputGate, cellWrite))
	cell := Must(Add(retain, write))
	hidden := Must(HadamardProd(outputGate, Must(Tanh(cell))))
	out = lstmout{
		hidden: hidden,
		cell:   cell,
	}
	return
}

type lstmout struct {
	hidden, cell *Node
}
