package main

import (
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var of = tensor.Float32

type FC struct {
	W   *Node
	Act func(x *Node) (*Node, error)
}

func (l *FC) fwd(x *Node) (*Node, error) {
	xw := Must(Mul(x, l.W))
	if l.Act == nil {
		return xw, nil
	}
	return l.Act(xw)
}

type NN struct {
	g *ExprGraph
	x *Node
	y *Node
	l []FC

	pred    *Node
	predVal Value
}

func NewNN(batchsize int) *NN {
	g := NewGraph()
	x := NewMatrix(g, of, WithShape(batchsize, 4), WithName("X"), WithInit(Zeroes()))
	y := NewVector(g, of, WithShape(batchsize), WithName("Y"), WithInit(Zeroes()))
	l := []FC{
		FC{W: NewMatrix(g, of, WithShape(4, 2), WithName("L0W"), WithInit(GlorotU(1.0))), Act: Tanh},
		FC{W: NewMatrix(g, of, WithShape(2, 128), WithName("L1W"), WithInit(GlorotU(1.0))), Act: Tanh},
		FC{W: NewMatrix(g, of, WithShape(128, 128), WithName("L2W"), WithInit(GlorotU(1.0))), Act: Tanh},
		FC{W: NewMatrix(g, of, WithShape(128, 1), WithName("L3W"), WithInit(GlorotU(1.0)))},
	}
	return &NN{
		g: g,
		x: x,
		y: y,
		l: l,
	}
}

func (nn *NN) learnables() Nodes {
	retVal := make(Nodes, 0, len(nn.l))
	for _, l := range nn.l {
		retVal = append(retVal, l.W)
	}
	return retVal
}

func (nn *NN) model() []ValueGrad { return NodesToValueGrads(nn.learnables()) }

func (nn *NN) cons() (pred *Node, err error) {
	pred = nn.x
	for _, l := range nn.l {
		if pred, err = l.fwd(pred); err != nil {
			return nil, err
		}
	}
	nn.pred = pred
	Read(nn.pred, &nn.predVal)

	cost := Must(Mean(Must(Square(Must(Sub(nn.y, pred))))))
	if _, err = Grad(cost, nn.learnables()...); err != nil {
		return nil, err
	}

	return pred, nil
}

type input struct {
	State  Point
	Action Vector
}

func (nn *NN) Let2(xs []input, y []float32) {
	xval := nn.x.Value().Data().([]float32)
	yval := nn.y.Value().Data().([]float32)
	// zero the data which may be contaminated by previous runs
	for i := range xval {
		xval[i] = 0
	}
	for i := range yval {
		yval[i] = 0
	}

	tmp := make([]float32, 0, len(xs)*4)
	for _, x := range xs {
		tmp = append(tmp, float32(x.State.X), float32(x.State.Y), float32(x.Action.X), float32(x.Action.Y))
	}
	copy(xval, tmp)
	copy(yval, y)
}

func (nn *NN) Let1(x input) {
	xval := nn.x.Value().Data().([]float32)
	// zero the data which may be contaminated by previous runs
	for i := range xval {
		xval[i] = 0
	}
	xval[0] = float32(x.State.X)
	xval[1] = float32(x.State.Y)
	xval[2] = float32(x.Action.X)
	xval[3] = float32(x.Action.Y)
}
