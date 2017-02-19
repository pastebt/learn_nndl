package main

import (
    "fmt"
    "math"
//    "sync"
//    "github.com/gonum/floats"
    "github.com/gonum/matrix/mat64"
)


type Cost interface {
    fn()
    delta()
}


type QuadraticCost struct {}
func (q QuadraticCost)fn(a, y *mat64.Dense) *mat64.Dense {return nil }
func (q QuadraticCost)delta(z, a, y *mat64.Dense) *mat64.Dense {
    m := mat64.NewDense(0, 0, nil)
    m.Sub(a, y)
    d := mat64.NewDense(0, 0, nil)
    d.MulElem(m, sigmoid_prime(z))
    return d
}


type CrossEntropyCost struct {}
func (c CrossEntropyCost)fn(a, y *mat64.Dense) *mat64.Dense {
    return nil
}
func (c CrossEntropyCost)delta(z, a, y *mat64.Dense) *mat64.Dense {
    m := mat64.NewDense(0, 0, nil)
    m.Sub(a, y)
    return m
}


type Network2 struct {
    num_layers int
    sizes   []int
    biases  []*mat64.Dense
    weights []*mat64.Dense
}


func NewNetwork2(sizes []int) *Network2 {
    nw := &Network2{num_layers: len(sizes), sizes: sizes}
    nw.biases = make([]*mat64.Dense, nw.num_layers - 1)
    nw.weights = make([]*mat64.Dense, nw.num_layers - 1)
    var d float64
    d = 1.0
    for i := 0; i < nw.num_layers - 1; i++ {
        y := nw.sizes[i + 1]
        x := nw.sizes[i]
        d = math.Sqrt(float64(x))
        nw.biases[i] = randyx(y, 1, 1)
        nw.weights[i] = randyx(y, x, d)
    }
    return nw
}


func (nw *Network2)feedforward(a *mat64.Dense) *mat64.Dense{
    for i, b := range nw.biases {
        a = sigmoid(add(npdot(nw.weights[i], a), b))
    }
    return a
}

func (nw *Network2)SGD(training_data []*ITEM, epochs, mini_batch_size int,
                      eta, lmbda float64, test_data []*ITEM) {
    for j := 0; j < epochs; j++ {
        shuffle(training_data)
        for k :=0; k < len(training_data); k = k + mini_batch_size {
            nw.update_mini_batch(training_data[k:k+mini_batch_size], eta,
                                 lmbda / float64(len(training_data)))
        }
        if test_data != nil {
            fmt.Printf("Epoch %02d: %d / %d\n",
                       j, nw.evaluate(test_data), len(test_data))
        } else {
            fmt.Printf("Epoch %02d complete\n", j)
        }
    }
}


func (nw *Network2)mb_add(ns, dn []*mat64.Dense) []*mat64.Dense {
    z := make([]*mat64.Dense, len(ns))
    for i := range ns {
        z[i] = add(ns[i], dn[i])
    }
    return z
}

func (nw *Network2)mb_cal(q, p float64, ns, dn []*mat64.Dense) []*mat64.Dense {
    z := make([]*mat64.Dense, len(ns))
    for i := range dn {
        m := mat64.DenseCopyOf(ns[i])
        fs := m.RawMatrix().Data
        ds := dn[i].RawMatrix().Data
        for j, f := range fs {
            fs[j] = q * f - p * ds[j]
        }
        z[i] = m
    }
    return z
}


func (nw *Network2)update_mini_batch(mini_batch []*ITEM, eta, lmnda_n float64) {
    nabla_b := zeros(nw.biases)
    nabla_w := zeros(nw.weights)
    for _, it  := range mini_batch {
        delta_nabla_b, delta_nabla_w := nw.backprop(it)
        nabla_b = nw.mb_add(nabla_b, delta_nabla_b)
        nabla_w = nw.mb_add(nabla_w, delta_nabla_w)
    }
    p := eta / float64(len(mini_batch))
    q := 1 - eta * lmnda_n  //1-eta*(lmbda/n)
    nw.weights = nw.mb_cal(q, p, nw.weights, nabla_w)
    nw.biases = nw.mb_cal(1, p, nw.biases, nabla_b)
}


func (nw *Network2)backprop(it *ITEM) (nabla_b, nabla_w []*mat64.Dense) {
    nabla_b = zeros(nw.biases)
    nabla_w = zeros(nw.weights)
    // feedforward
    activation := it.x
    // list to store all the activations, layer by layer
    activations := []*mat64.Dense{it.x}
    // list to store all the z vectors, layer by layer
    zs := make([]*mat64.Dense, 0, 10)
    for i := range nw.biases {
        z := mat64.NewDense(0, 0, nil)
        z.Add(npdot(nw.weights[i], activation), nw.biases[i])
        zs = append(zs, z)
        activation = sigmoid(z)
        activations = append(activations, activation)
    }
    // backward pass
    //delta := mat64.NewDense(0, 0, nil)
    //delta.MulElem(nw.cost_derivative(activations[len(activations) - 1], it.yv),
    //              sigmoid_prime(zs[len(zs) - 1]))
    cost := new(CrossEntropyCost)
    delta := cost.delta(zs[len(zs) - 1], activations[len(activations) - 1], it.yv)

    nabla_b[len(nabla_b) - 1] = delta
    nabla_w[len(nabla_w) - 1] = npdot(delta, activations[len(activations) - 2].T())
    for l := 2; l < nw.num_layers; l++ {
        z := zs[len(zs) - l]
        sp := sigmoid_prime(z)
        d := mat64.NewDense(0, 0, nil)
        d.MulElem(npdot(nw.weights[len(nw.weights) - l + 1].T(), delta), sp)
        delta = d
        nabla_b[len(nabla_b)-l] = delta
        nabla_w[len(nabla_w)-l] = npdot(delta, activations[len(activations)-l-1].T())
    }
    return
}

func (nw *Network2)evaluate(test_data []*ITEM) int {
    eq := 0
    for _, item := range test_data {
        if item.yi == argmax(nw.feedforward(item.x)) { eq = eq + 1 }
    }
    return eq
}

/*
func (nw *Network2)cost_derivative(output_activations, y *mat64.Dense) *mat64.Dense{
    m := mat64.NewDense(0, 0, nil)
    m.Sub(output_activations, y)
    return m
}
*/
