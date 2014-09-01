(ns pperceptrons.iris-pp-tests
  (:require [clojure.test :refer :all]
            [pperceptrons.core :refer :all]
            [pperceptrons.core-test :as ct]))



(def iris-data-vec
  (let [iris-rawdata  (slurp "./test/resources/iris.data")
        iris-lsplit   (clojure.string/split-lines iris-rawdata)
        iris-rsplit   (map (fn [x] (clojure.string/split x #",")) iris-lsplit)
        iris-in-out-form (map (fn [x]  [[(. Double parseDouble (nth x 0 ))
                                         (. Double parseDouble (nth x 1 ))
                                         (. Double parseDouble (nth x 2 ))
                                         (. Double parseDouble (nth x 3 ))]
                                        (if (= (nth x 4) "Iris-setosa")
                                               1.0 -1.0)
                                        (if (= (nth x 4) "Iris-virginica")
                                               1.0 -1.0)
                                        (if (= (nth x 4) "Iris-versicolor")
                                               1.0 -1.0)
                                        ])    iris-rsplit)
        ]

    iris-in-out-form
;    (shuffle iris-in-out-form)
))



(def flower-pp-t
  ;;We will train this same pp to recognise individual flowers
  (make-resonable-pp 4 0.501 false :seed 42 :size-boost 1
                                            :eta--auto-tune? true
                                            :gamma--tunning-rate 1.0))

(def pp-iris-setosa-t      (ct/test-trainging flower-pp-t iris-data-vec 30))
(def pp-iris-virginica-t   (ct/test-trainging flower-pp-t  (map (fn [x] [(first x) (nth x 2)])  iris-data-vec)   30))
(def pp-iris-versicolor-t  (ct/test-trainging flower-pp-t  (map (fn [x] [(first x) (nth x 3)])  iris-data-vec)    30))


(deftest testing-iris-acuracy
  (is (< 0.9 (:correctness pp-iris-setosa-t)))
  (is (< 0.9 (:correctness pp-iris-virginica-t)))
  (is (< 0.9 (:correctness pp-iris-versicolor-t)))

  )





