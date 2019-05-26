package main

import (
    "image/color"
    "math"

    "github.com/pkg/errors"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/palette/moreland"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
    "gorgonia.org/tensor"
)

type heatmap struct {
    x mat.Matrix
}

func (m heatmap) Dims() (c, r int)   { r, c = m.x.Dims(); return c, r }
func (m heatmap) Z(c, r int) float64 { return m.x.At(r, c) }
func (m heatmap) X(c int) float64    { return float64(c) }
func (m heatmap) Y(r int) float64    { return float64(r) }

type ticks []string

func (t ticks) Ticks(min, max float64) []plot.Tick {
    var retVal []plot.Tick
    for i := math.Trunc(min); i <= max; i++ {
        retVal = append(retVal, plot.Tick{Value: i, Label: t[int(i)]})
    }
    return retVal
}

func Heatmap(a *tensor.Dense) (p *plot.Plot, H, W vg.Length, err error) {
    switch a.Dims() {
    case 1:
        original := a.Shape()
        a.Reshape(original[0], 1)
        defer a.Reshape(original...)
    case 2:
    default:
        return nil, 0, 0, errors.Errorf("Can't do a tensor with shape %v", a.Shape())
    }

    m, err := tensor.ToMat64(a, tensor.UseUnsafe())
    if err != nil {
        return nil, 0, 0, err
    }

    pal := moreland.ExtendedBlackBody().Palette(256)
    // lum, _ := moreland.NewLuminance([]color.Color{color.Gray{0}, color.Gray{255}})
    // pal := lum.Palette(256)

    hm := plotter.NewHeatMap(heatmap{m}, pal)
    if p, err = plot.New(); err != nil {
        return nil, 0, 0, err
    }
    hm.NaN = color.RGBA{0, 0, 0, 0} // black
    p.Add(hm)

    sh := a.Shape()
    H = vg.Length(sh[0])*vg.Centimeter + vg.Centimeter
    W = vg.Length(sh[1])*vg.Centimeter + vg.Centimeter
    return p, H, W, nil
}

func Avg(a []float64) (retVal float64) {
    for _, v := range a {
        retVal += v
    }

    return retVal / float64(len(a))
}


