package main

import (
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"

	_ "net/http/pprof"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"
)

var (
	epochs     = flag.Int("epochs", 100, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "./mnist/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type nn struct {
	g          *gorgonia.ExprGraph
	w0, w1     *gorgonia.Node
	w5, w6, w7 *gorgonia.Node

	estMean   *gorgonia.Node // mean
	estSd     *gorgonia.Node // standard deviation stored in log scale
	floatHalf *gorgonia.Node
	epsilon   *gorgonia.Node

	out     *gorgonia.Node
	outMean *gorgonia.Node
	outVar  *gorgonia.Node
	predVal gorgonia.Value
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func newNN(g *gorgonia.ExprGraph) *nn {
	// Create node for w/weight
	w0 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(784, 256), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(256, 128), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	w5 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(8, 128), gorgonia.WithName("w5"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w6 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128, 256), gorgonia.WithName("w6"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w7 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(256, 784), gorgonia.WithName("w7"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	estMean := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128, 8), gorgonia.WithName("estMean"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	estSd := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128, 8), gorgonia.WithName("estSd"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	floatHalf := gorgonia.NewScalar(g, dt, gorgonia.WithName("floatHalf"))
	gorgonia.Let(floatHalf, 0.5)

	epsilon := gorgonia.GaussianRandomNode(g, dt, 0, 1, 100, 8)

	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w5: w5,
		w6: w6,
		w7: w7,

		estMean:   estMean,
		estSd:     estSd,
		floatHalf: floatHalf,
		epsilon:   epsilon,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w5, m.w6, m.w7, m.estMean, m.estSd}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var c1, c2, c5, c6, c7 *gorgonia.Node
	var sz *gorgonia.Node
	var l0, l1, l2, l3, l4, l5, l6, l7 *gorgonia.Node

	// Set first layer to be copy of input
	l0 = x
	log.Printf("l0 shape %v", l0.Shape())

	// Encoding - Part 1
	if c1, err = gorgonia.Mul(l0, m.w0); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if l1, err = gorgonia.Rectify(c1); err != nil {
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	log.Printf("l1 shape %v", l1.Shape())

	if c2, err = gorgonia.Mul(l1, m.w1); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if l2, err = gorgonia.Rectify(c2); err != nil {
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	log.Printf("l2 shape %v", l2.Shape())

	if l3, err = gorgonia.Mul(l2, m.estMean); err != nil {
		return errors.Wrap(err, "Layer 3 Multiplication failed")
	}
	log.Printf("l3 shape %v", l3.Shape())
	if l4, err = gorgonia.HadamardProd(m.floatHalf, gorgonia.Must(gorgonia.Mul(l2, m.estSd))); err != nil {
		return errors.Wrap(err, "Layer 4 Multiplication failed")
	}
	log.Printf("l4 shape %v", l4.Shape())

	// Sampling - Part 2
	if sz, err = gorgonia.Add(l3, gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Exp(l4)), m.epsilon))); err != nil {
		return errors.Wrap(err, "Layer Sampling failed")
	}
	log.Printf("sz shape %v", sz.Shape())

	// Decoding - Part 3
	if c5, err = gorgonia.Mul(sz, m.w5); err != nil {
		return errors.Wrap(err, "Layer 5 Convolution failed")
	}
	if l5, err = gorgonia.Rectify(c5); err != nil {
		return errors.Wrap(err, "Layer 5 activation failed")
	}
	log.Printf("l6 shape %v", l1.Shape())

	if c6, err = gorgonia.Mul(l5, m.w6); err != nil {
		return errors.Wrap(err, "Layer 6 Convolution failed")
	}
	if l6, err = gorgonia.Rectify(c6); err != nil {
		return errors.Wrap(err, "Layer 6 activation failed")
	}
	log.Printf("l6 shape %v", l6.Shape())

	if c7, err = gorgonia.Mul(l6, m.w7); err != nil {
		return errors.Wrap(err, "Layer 7 Convolution failed")
	}
	if l7, err = gorgonia.Sigmoid(c7); err != nil {
		return errors.Wrap(err, "Layer 7 activation failed")
	}
	log.Printf("l7 shape %v", l7.Shape())

	m.out = l7
	m.outMean = l3
	m.outVar = l4
	gorgonia.Read(l7, &m.predVal)
	return

}

const pixelRange = 255

func reversePixelWeight(px float64) byte {
	// return byte((pixelRange*px - pixelRange) / 0.9)
	return byte(pixelRange*math.Min(0.99, math.Max(0.01, px)) - pixelRange)
}

func visualizeRow(x []float64) *image.Gray {
	// since this is a square, we can take advantage of that
	l := len(x)
	side := int(math.Sqrt(float64(l)))
	r := image.Rect(0, 0, side, side)
	img := image.NewGray(r)

	pix := make([]byte, l)
	for i, px := range x {
		pix[i] = reversePixelWeight(px)
	}
	img.Pix = pix

	return img
}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(7945)

	// // intercept Ctrl+C
	// sigChan := make(chan os.Signal, 1)
	// signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// doneChan := make(chan bool, 1)

	var inputs tensor.Tensor
	var err error

	// load our data set
	trainOn := *dataset
	if inputs, _, err = mnist.Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples := inputs.Shape()[0]
	bs := *batchsize

	// if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
	// 	log.Fatal(err)
	// }

	// MNIST data consists of 28 by 28 black and white images
	// however we've imported it directly now as 784 different pixels
	// as a result, we need to reshape it to match what we actually want
	// if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
	// 	log.Fatal(err)
	// }

	// we should now also proceed to put in our desired variables
	// x is where our input should go, while y is the desired output
	g := gorgonia.NewGraph()
	// x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	x := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("y"))

	ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)

	m := newNN(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	valueOne := gorgonia.NewScalar(g, dt, gorgonia.WithName("valueOne"))
	valueTwo := gorgonia.NewScalar(g, dt, gorgonia.WithName("valueTwo"))
	gorgonia.Let(valueOne, 1.0)
	gorgonia.Let(valueTwo, 2.0)

	ioutil.WriteFile("simple_graph_2.dot", []byte(g.ToDot()), 0644)
	klLoss, err := gorgonia.Div(
		gorgonia.Must(gorgonia.Sum(
			gorgonia.Must(gorgonia.Sub(
				gorgonia.Must(gorgonia.Add(
					gorgonia.Must(gorgonia.Square(m.outMean)),
					gorgonia.Must(gorgonia.Exp(m.outVar)))),
				gorgonia.Must(gorgonia.Add(m.outVar, valueOne)))))),
		valueTwo)
	if err != nil {
		log.Fatal(err)
	}

	valueLoss, err := gorgonia.Sum(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, m.out)))))
	if err != nil {
		log.Fatal(err)
	}
	// valueCost := gorgonia.Must(gorgonia.Mean(value_losses))
	// cost = gorgonia.Must(gorgonia.Neg(cost))

	vaeCost := gorgonia.Must(gorgonia.Add(klLoss, valueLoss))

	ioutil.WriteFile("simple_graph_3.dot", []byte(g.ToDot()), 0644)

	// we wanna track costs
	var costVal gorgonia.Value
	gorgonia.Read(vaeCost, &costVal)

	if _, err = gorgonia.Grad(vaeCost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	log.Printf("learnables")

	// logger := log.New(os.Stderr, "", 0)
	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...), gorgonia.WithLogger(logger), gorgonia.WithWatchlist(), gorgonia.WithValueFmt("%1.1s"))

	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...), gorgonia.WithLogger(logger), gorgonia.TraceExec())
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(bs)), gorgonia.WithLearnRate(0.01))

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		// for i := 0; i < 10; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			// var xVal, yVal tensor.Tensor
			var xVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			// if yVal, err = inputs.Slice(sli{start, end}); err != nil {
			// 	log.Fatal("Unable to slice y")
			// }
			// if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
			// 	log.Fatal("Unable to reshape %v", err)
			// }
			if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
				log.Fatal("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, xVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}

			arrayOutput := m.predVal.Data().([]float64)
			yOutput := tensor.New(tensor.WithShape(bs, 784), tensor.WithBacking(arrayOutput))

			for j := 0; j < 1; j++ {
				rowT, _ := yOutput.Slice(sli{j, j + 1})
				row := rowT.Data().([]float64)

				img := visualizeRow(row)

				f, _ := os.OpenFile(fmt.Sprintf("training/%d - %d - %d training.jpg", j, b, i), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
				jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
				f.Close()
			}

			// solver.Step(m.learnables())
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		bar.Update()
		log.Printf("Epoch %d | cost %v", i, costVal)
	}
	bar.Finish()

	log.Printf("Run Tests")

	// load our test set
	if inputs, _, err = mnist.Load("test", loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples = inputs.Shape()[0]
	bs = *batchsize
	batches = numExamples / bs

	bar = pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)
	bar.Prefix(fmt.Sprintf("Epoch Test"))
	bar.Set(0)
	bar.Start()
	for b := 0; b < batches; b++ {
		start := b * bs
		end := start + bs
		if start >= numExamples {
			break
		}
		if end > numExamples {
			end = numExamples
		}

		var xVal tensor.Tensor
		if xVal, err = inputs.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice x")
		}

		// if yVal, err = inputs.Slice(sli{start, end}); err != nil {
		// 	log.Fatal("Unable to slice y")
		// }
		// if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
		// 	log.Fatal("Unable to reshape %v", err)
		// }
		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Fatal("Unable to reshape %v", err)
		}

		gorgonia.Let(x, xVal)
		gorgonia.Let(y, xVal)
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at epoch test: %v", err)
		}

		for j := 0; j < xVal.Shape()[0]; j++ {
			rowT, _ := xVal.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, _ := os.OpenFile(fmt.Sprintf("images/%d - %d input.jpg", b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
			f.Close()
		}

		arrayOutput := m.predVal.Data().([]float64)
		yOutput := tensor.New(tensor.WithShape(bs, 784), tensor.WithBacking(arrayOutput))

		for j := 0; j < yOutput.Shape()[0]; j++ {
			rowT, _ := yOutput.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, _ := os.OpenFile(fmt.Sprintf("images/%d - %d output.jpg", b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
			f.Close()
		}

		vm.Reset()
		bar.Increment()
	}
	log.Printf("Epoch Test | cost %v", costVal)

}
