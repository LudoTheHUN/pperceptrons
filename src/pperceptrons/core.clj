(ns pperceptrons.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]
            [clojure.core.matrix.implementations :as imp]))


;(m/set-current-implementation :persistent-vector)
;(m/set-current-implementation :vectorz)


;;Resources
;https://github.com/mikera/core.matrix
;https://github.com/mikera/core.matrix/blob/master/src/main/clojure/clojure/core/matrix/examples.clj
;https://github.com/clojure-numerics/core.matrix.stats/blob/develop/src/main/clojure/clojure/core/matrix/random.clj


;http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf

(def a (m/array [-1 2 -0.1]))

(def b (m/array [0.1 0.3 -0.4]))

(class a)
(m/shape a)

(m/mmul a b)


(defn input->z--input-array [input-vector]
     (m/array (conj input-vector -1.0))
  )

(input->z--input-array [3 24])

(class -1.0)
(input->z--input-array [1.0 2.0])


;(m/set-current-implementation :persistent-vector)
;(m/set-current-implementation :vectorz)

(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/scalar (m/mmul a--perceptron-weight-vector z--input-vector))]    ;;had to add m/scalar here to allow other matrix implementations
       (if (pos? mmulresult) 1.0 -1.0))
  )

;; we will need to re-implement the above in the learning steps so that we don't compute the same values more then once

;;(perceptron-f a b)


;pperceptron is made of many a--perceptron-weight-vectors

(def pperceptron
  ;;a pp with 3 perceptrons, taken in 3dim input, -1 in last dim for the threshold.
  (m/matrix
   [[-4.0 0.3 -0.3 0.5 -0.1]
    [0.1 -0.2 -0.4 -0.6 0.5]
    [0.2 -0.1 -0.7 -0.6 0.2]
    [0.2 -1.1 -0.7 -0.6 0.2]]))

(class pperceptron)

(defn total-pperceptron [pperceptron z--input-vector]
  (let [perceptron_value_fn (fn [perceptron] (perceptron-f perceptron z--input-vector))]
    (reduce +
            (doall (map perceptron_value_fn  (m/slices pperceptron))))
  ))


(total-pperceptron pperceptron  (m/array [-0.2 -0.2 10.5 -1.0 1.1]))

(total-pperceptron [[-4.0 0.3 -0.3 0.5 -0.1]
    [0.1 -0.2 -0.4 -0.6 0.5]
    [0.2 -0.1 -0.7 -0.6 0.2]
    [0.2 -1.1 -0.7 -0.6 0.2]]  [-0.2 -0.2 10.5 -1.0 1.1])

(total-pperceptron pperceptron  [-0.2 -0.2 -10.5 -1.0 0.2])
(total-pperceptron pperceptron  [-0.2 -0.2 1.5 -1.0 0.06])


(defn sp--squashing-function [pp-total rho--squashing-parameter]
     (cond
         (> pp-total rho--squashing-parameter)
           1.0
         (< pp-total (- rho--squashing-parameter))
           -1.0
         :else
         (/ pp-total rho--squashing-parameter)
   ))


(defn b-sp--binary-squashing-function [pp-total]
    (if (pos? pp-total ) 1.0 -1.0))



(defn pp-output [pperceptron input rho--squashing-parameter]
              (sp--squashing-function (total-pperceptron pperceptron    (input->z--input-array input))   rho--squashing-parameter))


(pp-output [[-4.0 0.3 -0.3 0.5 -0.1]
    [0.1 -0.2 -0.4 -0.6 0.5]
    [0.2 -0.1 -0.7 -0.6 0.2]
    [0.2 -1.1 -0.7 -0.6 0.2]] [-0.2 0.2 0.3 1.0] 3)

(pp-output pperceptron [-0.2 0.2 0.3 1.0] 3)
(pp-output pperceptron [-10.2 0.2 0.6 1.0] 10)


(def epsilon 0.3)   ;;how accureate we want to be, must be > 0

(def rho--squashing-parameter 10.0)  ;; whole numbers, 1 means a binary pperceptron

(defn f-abs [n]
 (cond
   (neg? n) (- n)
   :else n))


(def eta--learning-rate 0.1)


(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/scalar (m/mmul a--perceptron-weight-vector z--input-vector))]
       (if (pos? mmulresult) 1.0 -1.0))
  )



(def input  [-1.1 -3 0.3 0.4])
(m-ops/* input eta--learning-rate )




(def gamma--margin-around-zero 0.1)


(defn scaling-to-one-fn [perceptron eta--learning-rate]
   (m-ops/* perceptron (* -1.0 eta--learning-rate (- (m/length-squared perceptron) 1.0)))
  )

(scaling-to-one-fn [0.2 0.0 0.4] 0.01)



;(defn perceptron-f-amount [a--perceptron-weight-vector z--input-vector]
;     (m/mmul a--perceptron-weight-vector z--input-vector))

;(defn perceptron-f-vote [mmulresult]
;     (if (pos? mmulresult) 1.0 -1.0))



(defn pdelta-update-with-margin [pperceptron matrix-implementation input target-output epsilon rho--squashing-parameter eta--learning-rate mu-zeromargin-importance gamma--margin-around-zero]
  (let [z--input-vector (input->z--input-array input)

       perceptron_value_fn (fn [perceptron] (m/scalar (m/mmul perceptron z--input-vector)))   ;;had to add m/scalar here to allow other matrix implementations
       per-perceptron-totals  (doall (map perceptron_value_fn  (m/slices pperceptron)))
       out  (sp--squashing-function (reduce + (doall (map #(if (pos? (m/scalar %)) 1.0 -1.0) per-perceptron-totals))) rho--squashing-parameter)
     ;  out-vs-train-abs (f-abs (- out target-output))
       ]
    (m/matrix matrix-implementation
              (doall (map  (fn [perceptron perceptron_value]
                       (cond
                         (and (> out (+ target-output epsilon)) (pos? perceptron_value))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0 eta--learning-rate))
                         (and (< out (- target-output epsilon)) (neg? perceptron_value))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector       eta--learning-rate))
                         (and (<= out (+ target-output epsilon)) (pos? perceptron_value) (< perceptron_value gamma--margin-around-zero))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector  mu-zeromargin-importance  eta--learning-rate))
                         (and (>= out (- target-output epsilon))  (neg? perceptron_value) (< (* -1.0 gamma--margin-around-zero) perceptron_value ))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0  mu-zeromargin-importance  eta--learning-rate))
                         :else
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) )))
                    pperceptron
                    per-perceptron-totals)))))



m/*matrix-implementation*
pperceptron
input
(pdelta-update-with-margin
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

;;#<Matrix [[-3.3824,0.25368,-0.25368,0.4228,-0.08456],[0.10018,-0.20036,-0.40072,-0.60108,0.5009],[0.20012000000000002,-0.10006000000000001,-0.7004199999999999,-0.60036,0.20012000000000002],[0.19772,-1.08746,-0.69202,-0.59316,0.19772]]>

(def a_pp-learnt (last (take 1000
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

(pp-output a_pp-learnt  input 1)

(time
(pp-output
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
 input 1))


;-0.5
[[-0.9306841796056917 0.492038365677727 -0.12219569221147873 0.08037859586371489 0.10994474734168243]
 [0.13141469364693065 -0.15104593538009983 -0.4375218221172563 -0.655207892292099 0.5603377886938977]
 [0.20539833798013166 -0.10269916899006583 -0.7188941829304606 -0.6161950139403952 0.20539833798013166]
 [0.2877169781503483 -0.5213458925113474 -0.5907778124797812 -0.5231307451455035 0.27599214403636396]]

;-1.0
[[-0.9306841796056917 0.492038365677727 -0.12219569221147873 0.08037859586371489 0.10994474734168243]
 [0.13141469364693065 -0.15104593538009983 -0.4375218221172563 -0.655207892292099 0.5603377886938977]
 [0.20539833798013166 -0.10269916899006583 -0.7188941829304606 -0.6161950139403952 0.20539833798013166]
 [0.3970897308852651 -0.2239405292496575 -0.620953492717377 -0.5631859138938098 0.3754316867900243]]


[[-4.0 0.3 -0.3 0.5] [0.1 -0.2 -0.4 -0.6] [0.2 -0.1 -0.7 -0.6]]




;;TODO think about dynamics of eta--learning-rate and gamma--margin-around-zero
  ;these may need a dynamic update schedule.
  ;will need container for all the other aspects of the pperceptron
  ;create them with protocols / records ?



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




(defprotocol PPperceptron
  "Protocol for working with paralel perceptrons"
  (read-out     [pp input]  "returns the output the pperceptron for the given input")
  (train        [pp input output] "trains the paralel perceptron on one input-output example")
  (train-seq    [pp input-output-seq] "trains the paralel perceptron on a sequence of input examples with output values, shaped as [[inputs output]...]")
  (anneal-eta   [pp] "anneal eta, the learning rate")
  )


(defrecord pperceptron-record
  [pperceptron               ;; pperceptron ; the matrix holding the paralel perceptron weights which is as wide as the input +1 and as high as the number of perceptrons, n.
   matrix-implementation     ;; As supported by core.matrix, tested against  :persistent-vector and :vectorz
   n                         ;; n ; the total number of perceptrons in the pperceptron
   pwidth                    ;; size of each perceptron , width of pperceptron, +1 to size of input
   eta--learning-rate        ;; eta--learning-rate ; The learing rate. Typically 0.01 or less. Should be annealed.
   epsilon                   ;; epsilon ; how accureate we want to be, must be > 0
   rho--squashing-parameter  ;; rho--squashing-parameter ; An int. If set to 1, will force the pp to have binary output (-1,+1) in n is odd. Can be at most n. Typically set to  (/ 1 (* 2 epsilon))
   mu-zeromargin-importance  ;; mu-zeromargin-importance ; The zero margin parameter. Typically 1.
   gamma--margin-around-zero ;; gamma--margin-around-zero ; Margin around zero of the perceptron. Needs to be controlled for best performance, else set between 0.1 to 0.5
   ]
)


;;TODO hookup training function
(extend-protocol PPperceptron
  pperceptron-record
  (read-out [pp input]
     (pp-output (:pperceptron pp) input (:rho--squashing-parameter pp)))
  (train [pp input output]
      (assoc pp :pperceptron
        (pdelta-update-with-margin
           (:pperceptron pp)
           (:matrix-implementation pp)
           input
           output ;; target-output
           (:epsilon pp)
           (:rho--squashing-parameter pp)
           (:eta--learning-rate pp)
           (:mu-zeromargin-importance pp)
           (:gamma--margin-around-zero pp)
          )))
  (train-seq [pp input-output-seq] :WIP)
  (anneal-eta [pp] (assoc-in pp [:eta--learning-rate] 99 ))
  )


(defn uniform-dist-matrix-center-0
  "Returns an array of random samples from a uniform distribution on [0,1)
   Size may be either a number of samples or a shape vector."
  ([size seed] (uniform-dist-matrix-center-0 (m/current-implementation) size seed))
  ([matrix-implementation size seed]
    (let [size (if (number? size) [size] size)
          rnd  (java.util.Random. seed)]
      (m/compute-matrix matrix-implementation size
        (fn [& ixs]
          (- (* 2.0 (.nextDouble rnd)) 1.0))))))

;;(uniform-dist-matrix-center-0 :vectorz [3 3] 42)
;;(imp/get-implementation-key (uniform-dist-matrix-center-0 :vectorz [3] 42))

;;TODO DONE  constructing close to length one perceptrons from the start...
;;TODO take the random output and assuming it's a pperceptron, scale each perceptron towards length 1.0
(defn scale-to-size-one
 ([pperceptron] (scale-to-size-one (imp/get-implementation-key pperceptron) pperceptron ))
 ([matrix-implementation pperceptron]
   (m/matrix matrix-implementation (map (fn [perceptron] (m/div perceptron (m/length perceptron)))   (m/slices pperceptron)) )))

#_(time (map m/length (m/slices (scale-to-size-one (uniform-dist-matrix-center-0 :vectorz [5 100] 42)))))



(defn make-resonable-pp [inputsize  ;;do a (count input)
                         epsilon    ;;less then 0.5, nice numers here are eg: 0.25, 0.1
                         zerod?     ;;if true, n, number of perceptrons, will be even hence zero will be a possible output.
                         seed       ;;some int of your choosing
                         ]
 (let [matrix-implementation  :vectorz
       pwidth      (+ inputsize 1)
       n-prez (int (/ 2 epsilon))
       n     (cond (and zerod? (even? n-prez))
                     n-prez
                   (and zerod? (odd? n-prez))
                     (+ 1 n-prez)  ;to make it even
                   (and (not zerod?) (odd? n-prez))
                     n-prez
                   (and (not zerod?) (even? n-prez))
                     (+ 1 n-prez)
                   :else :this_should_never_happen!)
       rho-wip  (int (/ 1 (* 2 epsilon)))
       rho      (if (= rho-wip 0) 1 rho-wip)]
  (new pperceptron-record
   (scale-to-size-one (uniform-dist-matrix-center-0 matrix-implementation [n pwidth] seed))   ;; pperceptron ; the matrix holding the paralel perceptron weights which is as wide as the input +1 and as high as the number of perceptrons, n.
   matrix-implementation     ;; As supported by core.matrix, tested against  :persistent-vector and :vectorz
   n                         ;; n ; the total number of perceptrons in the pperceptron
   pwidth                    ;; size of each perceptron , width of pperceptron, +1 to size of input
   0.01                      ;; eta--learning-rate ; The learing rate. Typically 0.01 or less. Should be annealed or be dynamicall updated based on error function
   epsilon                   ;; epsilon ; how accureate we want to be, must be > 0
   rho                       ;; rho--squashing-parameter ; An int. If set to 1, will force the pp to have binary output (-1,+1) in n is odd. Can be at most n. Typically set to  (/ 1 (* 2 epsilon))
   1.0                       ;; mu-zeromargin-importance ; The zero margin parameter. Typically 1.
   0.5                       ;; gamma--margin-around-zero ; Margin around zero of the perceptron. Needs to be controlled for best performance, else set between 0.1 to 0.5
  )))





(def pp-a (make-resonable-pp 1 1.0 false 42))

(make-resonable-pp 1 1.0 true 42)

(read-out (make-resonable-pp 2 0.27 true 42) [2 1])
(train (make-resonable-pp 2 0.27 true 42) [2 1] 1)

(time
(read-out
(reduce (fn [xs [in out]] (train xs in out)) (make-resonable-pp 2 0.27 true 42) (repeat 100 [[2 1] 1])  )
[2 1])
)

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

(time (:pperceptron (make-resonable-pp 10 0.01 true 42)))

(time (pp-output (:pperceptron (make-resonable-pp 2 0.001 true 42)) [-0.21 0.1] 1000))

(m/shape (:pperceptron (make-resonable-pp 2 0.00001 true 42)))

(:n (make-resonable-pp 10 0.1 true 42))
(:pwidth (make-resonable-pp 10 0.1 true 42))
(:rho--squashing-parameter (make-resonable-pp 10 0.1 true 42))
(pp-output (:pperceptron (make-resonable-pp 10 0.1 true 42)) [0 0 5.337  0 0 -10 10 0 0  0]     (:rho--squashing-parameter (make-resonable-pp 10 0.1 true 42)))


;;record and protocol 101

(defprotocol my-protocol
  (foo [this]))

(extend-protocol my-protocol
  nil
  (foo [this] "nillbaby"))



(defrecord constant-foo [value boom]
  my-protocol
    (foo [this] (str value this boom)))


(def a (constant-foo. 7 33))

(foo a)


(foo nil)





;;get-implementation-key

(let [imp :vectorz
      mtrx (m/matrix imp [[1 2 3] [4 5 6]])]
 (m/matrix imp
   (map (fn [row] (m/div row (m/length row))) (m/slices mtrx)))
  )










