package main

import (
    "testing"
    "github.com/gonum/matrix/mat64"
)


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

