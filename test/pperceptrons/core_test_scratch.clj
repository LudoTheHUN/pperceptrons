(ns pperceptrons.core-test-scratch
  (:require [clojure.test :refer :all]
            [pperceptrons.core :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]
            [clojure.core.matrix.implementations :as imp]))


(def a "example array"
  (m/array [-1 2 -0.1]))


(def b (m/array [0.1 0.3 -0.4]))


(class a)
(m/shape a)


(m/mmul a b)

(m/mmul [2 1] [3 4 ])

(input->z--input-array [3 24])


(class -1.0)
(input->z--input-array [1.0 2.0])


(perceptron-f a b)


(def pperceptron
  ;;a pp with 3 perceptrons, taken in 3dim input, -1 in last dim for the threshold.
  (m/matrix
   [[-4.0 0.3 -0.3 0.5 -0.1]
    [0.1 -0.2 -0.4 -0.6 0.5]
    [0.2 -0.1 -0.7 -0.6 0.2]
    [0.2 -1.1 -0.7 -0.6 0.2]]))


(total-pperceptron pperceptron  (m/array [-0.2 -0.2 10.5 -1.0 1.1]))

(total-pperceptron [[-4.0 0.3 -0.3 0.5 -0.1]
    [0.1 -0.2 -0.4 -0.6 0.5]
    [0.2 -0.1 -0.7 -0.6 0.2]
    [0.2 -1.1 -0.7 -0.6 0.2]]  [-0.2 -0.2 10.5 -1.0 1.1])

(total-pperceptron pperceptron  [-0.2 -0.2 -10.5 -1.0 0.2])
(total-pperceptron pperceptron  [-0.2 -0.2 1.5 -1.0 0.06])


(pp-output [[-4.0 0.3 -0.3 0.5 -0.1]
    [0.1 -0.2 -0.4 -0.6 0.5]
    [0.2 -0.1 -0.7 -0.6 0.2]
    [0.2 -1.1 -0.7 -0.6 0.2]] [-0.4 0.2 0.3 1.0] 2)

(pp-output pperceptron [-0.2 0.2 0.3 1.0] 3)
(pp-output pperceptron [-10.2 0.2 0.6 1.0] 10)





(def epsilon 0.3)   ;;how accureate we want to be, must be > 0

(def rho--squashing-parameter 10.0)  ;; whole numbers, 1 means a binary pperceptron

(def eta--learning-rate 0.1)

(def input  [-1.1 -3 0.3 0.4])
(m-ops/* input eta--learning-rate )

(def gamma--margin-around-zero 0.1)

(scaling-to-one-fn [0.2 0.0 0.4] 0.01)

;(defn perceptron-f-amount [a--perceptron-weight-vector z--input-vector]
;     (m/mmul a--perceptron-weight-vector z--input-vector))

;(defn perceptron-f-vote [mmulresult]
;     (if (pos? mmulresult) 1.0 -1.0))



m/*matrix-implementation*
pperceptron
input
#_(pdelta-update-with-margin
    pperceptron
    :vectorz
    input
    0.0   ;;; target-output
    0.01  ;;; epsilon
    3.0   ;;; rho--squashing-parameter
    0.01  ;;; eta--learning-rate
    0.1   ;;; mu-zeromargin-importance
    0.1   ;;; gamma--margin-around-zero
 )


#_(def a_pp-learnt (last (take 1000
      (iterate (fn [x]
(pdelta-update-with-margin
    x
    :vectorz
    input
    0.9   ;;; target-output
    0.25  ;;; epsilon
    1   ;;; rho--squashing-parameter
    0.01  ;;; eta--learning-rate
    1.0   ;;; mu-zeromargin-importance
    0.5   ;;; gamma--margin-around-zero
 )) pperceptron))))


#_(pp-output a_pp-learnt  input 1)




#_(pp-output
(last (take 1000
      (iterate (fn [x]
(pdelta-update-with-margin
    x
    :vectorz
    ;:persistent-vector
    input
    0.9   ;;; target-output
    0.25  ;;; epsilon
    1     ;;; rho--squashing-parameter
    0.01  ;;; eta--learning-rate
    1.0   ;;; mu-zeromargin-importance
    0.5   ;;; gamma--margin-around-zero
 )) pperceptron)))
 input 1)



(pp-output pperceptron input 3.0)

(m/join [1 2 3] [3 4 5])

(m/mmul [[0.1 0.2 0.3]
       [0.2 0.2 0.1]
       [0.3 0.1 -0.3]
       ]

      [1 2 3]
      )


(m/mmul [[0.1 0.2 0.3]
       [0.2 0.2 0.1]
       [0.3 0.1 -0.3]
       ]

      [[3 3 3]
       [2 2 2]
       [1 1 1]]
      )











(def pp-a
  (make-resonable-pp 1 0.5 false 42)
  )

(anneal-eta (make-resonable-pp 1 1.0 true 42))

(/ 2 50.0)
(read-out (make-resonable-pp 1 0.1 false 42 10) [1.1549])

(let [pp (make-resonable-pp 1 0.01 false 41 10)]
  (map  (fn [x] (read-out pp [x])) (range -2 2 0.1)))


(frequencies
(let [pp (make-resonable-pp 2 0.25 true 42 1)]
  (map  (fn [x] (read-out pp [x x])) (range -3 3 0.01))))




(frequencies
(let [pp (make-resonable-pp 2 0.25 true 42 1)]
  (map  (fn [x] (read-out pp [x x])) (range -3 3 0.1))))


(let [f (fn [x y] [x y])]
  (map f (map (fn [x] (fn [y] (f y x))) (range -3 3 0.1)) (range -1 1 0.1)))



(defn cover-2d-range [pp]
   (map  (fn [[x y]] (read-out pp [x y]))
         (apply concat (map (fn [x] (map (fn [y] [x y]) (range -2 2 0.1)) ) (range -2 2 0.1)))))

(defn frequencies-of-resonable-pp-2d [epsilon zerod? seed]
    (sort (frequencies (apply concat (map (fn [seed] (cover-2d-range (make-resonable-pp 2 epsilon zerod? seed))) (range 10))))))

(frequencies-of-resonable-pp-2d 0.1 true 42)


(read-out (make-resonable-pp 2 0.27 true 42) [2 1])
(train (make-resonable-pp 2 0.27 true 42) [2 1] 1)


(read-out
(reduce (fn [xs [in out]] (train xs in out)) (make-resonable-pp 2 0.27 true 42) (repeat 100 [[2 1] 1])  )
[2 1])


[0.6516616558174498,0.5246870281514043,-0.5477596268683069],
[-0.18593836403808622,0.5749282725464294,0.7967963387469613],
[-0.21376830945425768,-0.8922118302823092,-0.3978205102913648],
[0.5549199829079453,0.8225235171640101,-0.12457478188458596],
[0.5900107775778086,-0.26780756812858175,-0.7616865423486782],
[0.7468354261251801,-0.42599196260929023,0.5106541822885579],
[-0.6133673546743034,0.5202591780846183,0.5942313303544274],
[0.4679312985708796,0.3924670063715887,0.7918395979110127]


[0.6516616558174498,0.5246870281514043,-0.5477596268683069],

;(m/set-current-implementation :persistent-vector)
;(m/set-current-implementation :vectorz)
(class (:pperceptron (make-resonable-pp 10 0.01 true 42)))

(:pperceptron (make-resonable-pp 10 0.01 true 42))

(pp-output (:pperceptron (make-resonable-pp 2 0.001 true 42)) [-0.21 0.1] 1000)

(m/shape (:pperceptron (make-resonable-pp 2 0.00001 true 42)))

(:n (make-resonable-pp 10 0.1 true 42))
(:pwidth (make-resonable-pp 10 0.1 true 42))
(:rho--squashing-parameter (make-resonable-pp 10 0.1 true 42))
(pp-output (:pperceptron (make-resonable-pp 10 0.1 true 42)) [0 0 5.337  0 0 -10 10 0 0  0]     (:rho--squashing-parameter (make-resonable-pp 10 0.1 true 42)))


;;record and protocol 101

(defprotocol my-protocol
  (boo [this] "uses foo to do boo")
  (foo [this]))

(extend-protocol my-protocol
  nil
  (foo [this] "nillbaby"))



(defrecord constant-foo [value boom]
  my-protocol
    (boo [this] (count (foo this)))
    (foo [this] (str value this boom)))


(def a (constant-foo. 7 33))

(foo a)
(boo a)

(foo nil)





;;get-implementation-key

(let [imp :vectorz
      mtrx (m/matrix imp [[1 2 3] [4 5 6]])]
 (m/matrix imp
   (map (fn [row] (m/div row (m/length row))) (m/slices mtrx)))
  )

