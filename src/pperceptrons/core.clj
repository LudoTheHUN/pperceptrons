(ns pperceptrons.core
  (:require [clojure.core.matrix :as m]))


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


(defn input->z--input-vector [input-vector]
     (m/array (conj input-vector -1.0))
  )

(class -1.0)
(input->z--input-vector [1.0 2.0])


(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/mmul a--perceptron-weight-vector z--input-vector)]
       (if (pos? mmulresult) 1.0 -1.0))
  )

(perceptron-f a b)


;pperceptron is made of many a--perceptron-weight-vectors

(def pperceptron
  ;;a pp with 3 perceptrons, taken in 3dim input, -1 in last dim for the threshold.
  [[-4.0 0.3 -0.3 0.5]
   [0.1 -0.2 -0.4 -0.6]
   [0.2 -0.1 -0.7 -0.6]])



(defn total-pperceptron [pperceptron input-vector]
  (let [perceptron_value (fn [perceptron] (perceptron-f perceptron input-vector))]
    (reduce + (map perceptron_value  (m/slices pperceptron)))
  ))


(total-pperceptron pperceptron  [-0.2 -0.2 10.5 -1.0])
(total-pperceptron pperceptron  [-0.2 -0.2 -10.5 -1.0])
(total-pperceptron pperceptron  [-0.2 -0.2 1.5 -1.0])




(defn sp--squashing-function [pp-total q--squashing-parameter]
     (cond
         (> pp-total q--squashing-parameter)
           1.0
         (< pp-total (- q--squashing-parameter))
           -1.0
         :else
         (/ pp-total q--squashing-parameter)
   ))


(defn b-sp--binary-squashing-function [pp-total]
    (if (pos? pp-total ) 1.0 -1.0))



(defn pp-output [pperceptron input q--squashing-parameter]
              (sp--squashing-function (total-pperceptron pperceptron    (input->z--input-vector input))   q--squashing-parameter)
  )




(pp-output pperceptron [-0.2 0.2 0.3] 3)
(pp-output pperceptron [-10.2 0.2 0.6] 10)





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






