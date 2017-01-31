package main

import (
    "testing"
    "github.com/gonum/floats"
    "github.com/gonum/matrix/mat64"
)


func TestT(tst *testing.T) {
    z := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                        4, 5, 6})
    o := mat64.NewDense(3, 2, []float64{1, 4,
                                        2, 5,
                                        3, 6})
    s := T(z)
    if !mat64.Equal(s, o) {
        tst.Error("T wrong",  s, o)
    }
}

func TestSigmoid(tst *testing.T) {
    /*
    z = array([[1, 2, 3],
               [4, 5, 6]])
    1.0/(1.0+np.exp(-a)) = 
        array([[ 0.73105858,  0.88079708,  0.95257413],
               [ 0.98201379,  0.99330715,  0.99752738]])
    */
    z := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                        4, 5, 6})
    o := mat64.NewDense(2, 3, []float64{
                0.73105858,  0.88079708,  0.95257413,
                0.98201379,  0.99330715,  0.99752738})
    s := sigmoid(z)
    // 0.000000001 will failed
    if !mat64.EqualApprox(s, o, 0.00000001) {
        tst.Error("sigmoid Wrong")
    }
}


func TestDot(tst *testing.T) {
    w := mat64.NewDense(3, 2, []float64{1.682852  , -0.03961098,
                                        0.03793304,  0.42585802,
                                        0.74246289, -0.03209176})
    a := mat64.NewDense(2, 1, []float64{-0.1631285,
                                        -1.73508289})
    /* np.dot(w, a)
    array([[-0.20579278],
           [-0.74508692],
           [-0.06543499]]) */
    o := mat64.NewDense(3, 1, []float64{-0.20579278,
                                        -0.74508692,
                                        -0.06543499})
    d := dot(w, a)
    // 0.000000001 will failed
    if !mat64.EqualApprox(d, o, 0.00000001) {
        tst.Error("dot Wrong")
    }
/*
    a = mat64.NewDense(1, 3, []float64{1, 2, 3})
    b := mat64.NewDense(3, 1, []float64{7,
                                        8,
                                        9})
    o = 
*/
}


func TestArgmax(tst *testing.T) {
    /*
    a = np.array([[1, 2, 3],[40, 5, 6]])
    np.argmax(a) -> 3
    */
    a := mat64.NewDense(2, 3, []float64{1,  2, 3,
                                        40, 5, 6})
    o := argmax(a)
    if o != 3 {
        tst.Error("argmax Wrong, should 3, get", o)
    }
    a = mat64.NewDense(2, 3, []float64{10, 2, 3,
                                       4,  5, 6})
    o = argmax(a)
    if o != 0 {
        tst.Error("argmax Wrong, should 0, get", o)
    }
    a = mat64.NewDense(2, 3, []float64{1, 2, 30,
                                       4,  5, 6})
    o = argmax(a)
    if o != 2 {
        tst.Error("argmax Wrong, should 2, get", o)
    }
    a = mat64.NewDense(2, 3, []float64{1, 2,
                                       3, 4,
                                       5, 6})
    o = argmax(a)
    if o != 5 {
        tst.Error("argmax Wrong, should 5, get", o)
    }
    a = mat64.NewDense(1, 6, []float64{1, 2, 30, 4,  5, 6})
    o = argmax(a)
    if o != 2 {
        tst.Error("argmax Wrong, should 2, get", o)
    }
}


func TestSigmoid_prime(tst *testing.T) {
    /*
    a = np.array([[1, 2, 3],[4, 5, 6]])
    x = 1.0/(1.0+np.exp(-a))
    y = 1.0/(1.0+np.exp(-a))
    x
    array([[ 0.73105858,  0.88079708,  0.95257413],
           [ 0.98201379,  0.99330715,  0.99752738]])
    y
    array([[ 0.73105858,  0.88079708,  0.95257413],
           [ 0.98201379,  0.99330715,  0.99752738]])
    x * (1 - y)
    array([[ 0.19661193,  0.10499359,  0.04517666],
           [ 0.01766271,  0.00664806,  0.00246651]])
    */
    z := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                        4, 5, 6})
    o := mat64.NewDense(2, 3, []float64{
                0.19661193,  0.10499359,  0.04517666,
                0.01766271,  0.00664806,  0.00246651})
    s := sigmoid_prime(z)
    // 0.000000001 will failed
    if !mat64.EqualApprox(s, o, 0.00000001) {
        tst.Error("sigmoid_prime Wrong")
    }
}


func BenchmarkAdd0(bm *testing.B) {
    a := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                        4, 5, 6})
    b := mat64.NewDense(2, 3, []float64{9, 8, 7,
                                        4, 5, 6})
    z := mat64.NewDense(2, 3, []float64{0, 0, 0, 0, 0, 0})
    for i := 0; i < bm.N; i++ {
        z.Reset()
        z.Add(a, b)
    }
}


func BenchmarkAdd1(bm *testing.B) {
    a := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                        4, 5, 6})
    b := mat64.NewDense(2, 3, []float64{9, 8, 7,
                                        4, 5, 6})
    for i := 0; i < bm.N; i++ {
        z := mat64.NewDense(0, 0, nil)
        //z := mat64.DenseCopyOf (a)
        z.Add(a, b)
    }
}


func BenchmarkAdd2(bm *testing.B) {
    a := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                        4, 5, 6})
    b := mat64.NewDense(2, 3, []float64{9, 8, 7,
                                        4, 5, 6})
    z := mat64.NewDense(2, 3, []float64{0, 0, 0, 0, 0, 0})
    for i := 0; i < bm.N; i++ {
        //z := mat64.NewDense(0, 0, nil)
        z.Apply(func(x,y int, v float64)float64{
                    return b.At(x, y) + v
                }, a)
    }
}


func BenchmarkAdd3(bm *testing.B) {
    a := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                        4, 5, 6})
    b := mat64.NewDense(2, 3, []float64{9, 8, 7,
                                        4, 5, 6})
    for i := 0; i < bm.N; i++ {
        a.Apply(func(x,y int, v float64)float64{
                    return b.At(x, y) + v
                }, a)
    }
}


func TestAdd(tst *testing.T) {
    /*
    a = np.array([[1, 2, 3],[40, 5, 6]])
    b = np.array([[1],[2]])
    a + b 
    array([[ 2,  3,  4],
           [42,  7,  8]])
    b + a
    array([[ 2,  3,  4],
           [42,  7,  8]])
    b = np.array([[1,2]])
    a + b ValueError
    b = np.array([[1, 2, 3]])
    b + a
    array([[ 2,  4,  6],
           [41,  7,  9]])
    */
    a := mat64.NewDense(2, 3, []float64{1, 2, 3,
                                       40, 5, 6})
    b := mat64.NewDense(2, 1, []float64{1,
                                        2})
    o := mat64.NewDense(2, 3, []float64{2,  3,  4,
                                       42,  7,  8})
    s := add(a, b)
    if !mat64.Equal(o, s) {
        tst.Error("TestAdd error", s, o)
    }
    s = add(b, a)
    if !mat64.Equal(o, s) {
        tst.Error("TestAdd error", s, o)
    }
    b = mat64.NewDense(1, 2, []float64{1,2})
    s1, s2 := add(a, b), add(b, a)
    if s1 != nil || s2 != nil {
        tst.Error("TestAdd error", s, o)
    }

    b = mat64.NewDense(1, 3, []float64{1, 2, 3})
    o = mat64.NewDense(2, 3, []float64{2,  4,  6,
                                      41,  7,  9})
    s = add(b, a)
    if !mat64.Equal(o, s) {
        tst.Error("TestAdd error", s, o)
    }
    s = add(a, b)
    if !mat64.Equal(o, s) {
        tst.Error("TestAdd error", s, o)
    }
}


func TestCost_derivative(tst *testing.T) {
    /*
    a = np.array([[1, 2, 3, 4]])
    a - 1
        array([[0, 1, 2, 3]])
    */
    /*
    n := NewNetwork([]int{784, 30, 10})
    a := mat64.NewDense(1, 4, []float64{1, 2, 3, 4})
    o := mat64.NewDense(1, 4, []float64{0, 1, 2, 3})
    s := n.cost_derivative(a, 1)
    if !mat64.Equal(o, s) {
        tst.Error("TestCost_derivative error", s, o)
    }
    a = mat64.NewDense(3, 2, []float64{1, 2,
                                       3, 4,
                                       5, 7})
    o = mat64.NewDense(3, 2, []float64{0, 1,
                                       2, 3,
                                       4, 6})
    s = n.cost_derivative(a, 1)
    if !mat64.Equal(o, s) {
        tst.Error("TestCost_derivative error", s, o)
    }
    */
}


func make01(r, c int) *mat64.Dense {
    d := make([]float64, r * c)
    for i := range d { d[i] = 0.1 * float64(i + 1) }
    return mat64.NewDense(r, c, d)
}


func tNetwork(sizes []int) *Network {
    nw := &Network{num_layers: len(sizes), sizes: sizes}
    nw.biases = make([]*mat64.Dense, nw.num_layers - 1)
    nw.weights = make([]*mat64.Dense, nw.num_layers - 1)
    for i := 0; i < nw.num_layers - 1; i++ {
        y := nw.sizes[i + 1]
        x := nw.sizes[i]
        //nw.biases[i] = randyx(y, 1)
        nw.biases[i] = make01(y, 1)
        //nw.weights[i] = randyx(y, x)
        nw.weights[i] = make01(y, x)
    }
    return nw
}


func TestFeedforward(tst *testing.T) {
    nw := tNetwork([]int{4, 3, 2})
    o := mat64.NewDense(2, 1, []float64{0.66719738,
                                        0.84321898})
    f := nw.feedforward(mat64.NewDense(4, 1, []float64{1, 2, 3, 4}))
    // 0.000000001 will failed
    if !mat64.EqualApprox(f, o, 0.00000001) {
        tst.Error("feedforward Wrong", f, o)
    }
    f = nw.feedforward(mat64.NewDense(4, 1, []float64{5, 2, 3, 4}))
    if mat64.EqualApprox(f, o, 0.00000001) {
        tst.Error("feedforward2 Wrong", f, o)
    }
}


func TestBackprop(tst *testing.T) {
    /*
    b: [array([[-1.52167557e-02],
               [-3.97545878e-04],
               [-8.63189706e-06]]),
        array([[-0.96207731],
               [-0.68173021]])
        ]
    w: [array([[-1.52167557e-02, -3.04335114e-02, -4.56502670e-02, -6.08670227e-02],
               [-3.97545878e-04, -7.95091755e-04, -1.19263763e-03, -1.59018351e-03],
               [-8.63189706e-06, -1.72637941e-05, -2.58956912e-05, -3.45275882e-05]]),
        array([[-0.9206048 , -0.96135957, -0.96206541],
               [-0.65234269, -0.68122162, -0.68172178]])
        ]
    */
    b0 := []float64{-1.52167557e-02, -3.97545878e-04, -8.63189706e-06}
    b1 := mat64.NewDense(2, 1, []float64{-0.96207731, -0.68173021})
    w0 := mat64.NewDense(3, 4, []float64{-1.52167557e-02, -3.04335114e-02, -4.56502670e-02, -6.08670227e-02,
                                         -3.97545878e-04, -7.95091755e-04, -1.19263763e-03, -1.59018351e-03,
                                         -8.63189706e-06, -1.72637941e-05, -2.58956912e-05, -3.45275882e-05})
    w1 := mat64.NewDense(2, 3, []float64{-0.9206048 , -0.96135957, -0.96206541,
                                         -0.65234269, -0.68122162, -0.68172178})
    nw := tNetwork([]int{4, 3, 2})
    i := &ITEM{x: mat64.NewDense(4, 1, []float64{1, 2, 3, 4}),
               yv: mat64.NewDense(2, 1, []float64{5, 6}),
               yi: 1}
    b, w := nw.backprop(i)
    //tst.Log("b =", b[0])
    //tst.Log("w =", w[0])
    if len(b) != 2 || len(w) != 2 { tst.Error("backprop size Wrong") }
    if !floats.EqualApprox(b[0].RawMatrix().Data, b0, 0.00000001) {
        tst.Error("backprop b0 Wrong", b[0], b0)
    }
    if !mat64.EqualApprox(b[1], b1, 0.00000001) {
        tst.Error("backprop b1 Wrong", b[1], b1)
    }
    if !mat64.EqualApprox(w[0], w0, 0.00000001) {
        tst.Error("backprop w0 Wrong", w[0], w0)
    }
    if !mat64.EqualApprox(w[1], w1, 0.00000001) {
        tst.Error("backprop w1 Wrong", w[1], w1)
    }
}


func TestUpdate_mini_batch(tst *testing.T) {
    /*
    biases: [array([[ 0.07777175],
                    [ 0.17329118],
                    [ 0.27326081]]),
             array([[-0.01917578],
                    [ 0.00533144]])]
    weights:[array([[ 0.07737407,  0.20039768,  0.3       ,  0.4       ],
                    [ 0.47756766,  0.59572352,  0.7       ,  0.8       ],
                    [ 0.88026383,  0.99299698,  1.1       ,  1.2       ]]),
             array([[ 0.02724625,  0.11397004,  0.20332693],
                    [ 0.2948714 ,  0.37161299,  0.45173367]])]
    */
    b0 := mat64.NewDense(3, 1, []float64{0.07777175,
                                         0.17329118,
                                         0.27326081})
    b1 := mat64.NewDense(2, 1, []float64{-0.01917578,
                                         0.00533144})
    w0 := mat64.NewDense(3, 4, []float64{0.07737407,  0.20039768,  0.3       ,  0.4,
                                         0.47756766,  0.59572352,  0.7       ,  0.8,
                                         0.88026383,  0.99299698,  1.1       ,  1.2})
    w1 := mat64.NewDense(2, 3, []float64{0.02724625,  0.11397004,  0.20332693,
                                         0.2948714 ,  0.37161299,  0.45173367})
    its := []*ITEM{
                &ITEM{
                    x: mat64.NewDense(4, 1, []float64{0, 1, 0, 0}),
                    yv: mat64.NewDense(2, 1, []float64{0, 1}),
                    yi: 1,
                    },
                &ITEM{
                    x: mat64.NewDense(4, 1, []float64{1, 0, 0, 0}),
                    yv: mat64.NewDense(2, 1, []float64{1, 0}),
                    yi: 0,
                    },
            }
    nw := tNetwork([]int{4, 3, 2})
    nw.update_mini_batch(its, 4.0)
    if len(nw.biases) != 2 || len(nw.weights) != 2 { tst.Error("update_mini_batch size Wrong") }
    if !mat64.EqualApprox(nw.biases[0], b0, 0.00000001) {
        tst.Error("backprop b0 Wrong", nw.biases[0], b0)
    }
    if !mat64.EqualApprox(nw.biases[1], b1, 0.00000001) {
        tst.Error("backprop b1 Wrong", nw.biases[1], b1)
    }
    if !mat64.EqualApprox(nw.weights[0], w0, 0.00000001) {
        tst.Error("backprop w0 Wrong", nw.weights[0], w0)
    }
    if !mat64.EqualApprox(nw.weights[1], w1, 0.00000001) {
        tst.Error("backprop w1 Wrong", nw.weights[1], w1)
    }

    //argmax(nw.feedforward(item.x))
}
