package main

import (
    "os"
    "log"
    "math"
    "math/rand"
    "bufio"
    "strconv"
    "strings"
    "github.com/gonum/matrix/mat64"
)


func load_one(infn string) (err error) {
    fin, err := os.Open(infn)
    if err != nil {
        log.Fatal(err)
    }
    defer fin.Close()
    println(infn)
    scanner := bufio.NewScanner(fin)
    for scanner.Scan() {
        cols := strings.Fields(scanner.Text())
        if len(cols) != 785 { log.Fatalf("got column %d", len(cols)) }
        var n int64
        n, err = strconv.ParseInt(cols[0], 10, 32)
        if err != nil { log.Fatal(cols[0], err) }
        dx := make([]float64, 784)
        var f float64
        for i := 1; i < 785; i++ {
            f, err = strconv.ParseFloat(cols[i], 32)
            if err != nil { log.Fatal(cols[i], err) }
            dx[i - 1] = f
        }
        m := mat64.NewDense(len(dx), 1, dx)
        if n > 9 || f > 10000 || m == nil { log.Fatalf("n=%d, f=%f, m=%#v", n, f, m) }
    }
    return
}


// The sigmoid function.
// Scale, Inverse, Apply
func sigmoid(z *mat64.Dense) *mat64.Dense {
    //return 1.0 / (1.0 + math.Exp(-z))
    a := mat64.NewDense(0, 0, nil)
    e := mat64.NewDense(0, 0, nil)
    o := mat64.NewDense(0, 0, nil)
    i := mat64.NewDense(0, 0, nil)
    a.Scale(-1, z)
    e.Exp(a)
    o.Apply(func(x, y int, v float64)float64{return 1.0 + v}, e)
    i.Inverse(o)
    return i
}


// Derivative of the sigmoid function.
func sigmoid_prime(z *mat64.Dense) *mat64.Dense {
    //return sigmoid(z) * (1 - sigmoid(z))
    m := mat64.NewDense(0, 0, nil)
    a := mat64.NewDense(0, 0, nil)
    s := sigmoid(z)
    a.Apply(func(x, y int, v float64)float64{return 1.0 - v}, s)
    m.Mul(s, a)
    return m
}


// The sigmoid function.
func sigmoid_(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}


// Derivative of the sigmoid function.
func sigmoid_prime_(z float64) float64 {
    return sigmoid_(z) * (1 - sigmoid_(z))
}


func randyx(y, x int) *mat64.Dense {
    z := x * y
    dat := make([]float64, z)
    for i := 0; i < z; i++ {
        dat[i] = rand.Float64()
    }
    return mat64.NewDense(y, x, dat)
}


type Network struct {
    num_layers int
    sizes   []int
    biases  []*mat64.Dense
    weights []*mat64.Dense
}


func NewNetwork(sizes []int) *Network {
    nw := &Network{num_layers: len(sizes), sizes: sizes}
    nw.biases = make([]*mat64.Dense, nw.num_layers - 1)
    nw.weights = make([]*mat64.Dense, nw.num_layers - 1)
    for i := 0; i < nw.num_layers - 1; i++ {
        y := nw.sizes[i + 1]
        x := nw.sizes[i]
        nw.biases[i] = randyx(y, 1)
        nw.weights[i] = randyx(y, x)
    }
    return nw
}


func (nw *Network)feedforward () {}
func (nw *Network)SGD() {}
func (nw *Network)update_mini_batch() {}
func (nw *Network)backprop() {}
func (nw *Network)evaluate() {}
func (nw *Network)cost_derivative() {}


func main() {
    load_one("trai_data.txt")
    load_one("vali_data.txt")
    load_one("test_data.txt")
}
