(ns clj-elm.data-test
  (:require [clj-elm.data :refer :all]
            [midje.sweet :refer :all]
            [midje.repl :refer (autotest load-facts)]
            [incanter.core :as c :exclude [update]]
            [incanter.io :as io]
            [incanter.stats :as st]
            [svm.core :as svm])
  (:import [clj_elm.data DataSet]))

(def australian
  (io/read-dataset "data/australian.csv" :delim \, :header true))

(def australian-data
  (read-dataset "data/australian.csv" 14 :header true))

(def libsvmlian
  (read-dataset-lib-svm "data/australian"))

(def A [[1 1 1 1] [1 1 1 1] [1 1 1 1]])
(def B [[10 20 0] [0 40 0] [20 50 0] [-10 10 0]])

(facts "test-parse-lib-svm-data"
  (fact "(parse-lib-svm-data line)"
    (parse-lib-svm-data (first (svm/read-dataset "data/australian")))
    => [-1 [1.0 22.08 11.46 2.0 4.0 4.0 1.585 0.0 0.0 0.0 1.0 2.0 100.0 1213.0]]))

(facts "test-read-dataset-lib-svm"
  (facts "(read-dataset-lib-svm path)"
    (take 5 (:classes (read-dataset-lib-svm "data/australian")))
    => [-1 -1 -1 1 1]
    (take 5 (first (:features (read-dataset-lib-svm "data/australian"))))
    => [1.0 22.08 11.46 2.0 4.0]))

(facts "test-num-of-feature"
  (facts "(num-of-feature dataset)"
    (num-of-feature australian-data) => 14))

(facts "test-data-set"
  (facts "(data-set data cidx)"
    (take 5 (:classes australian-data))
    => [-1 -1 -1 1 1]
    (take 5 (:classes libsvmlian))
    => [-1 -1 -1 1 1]
    (take 5 (first (:features australian-data)))
    => [1 22.08 11.46 2 4]
    (take 5 (first (:features libsvmlian)))
    => [1.0 22.08 11.46 2.0 4.0]))

(facts "test-concat-dataset"
  (facts "(concat-dataset dsa dsb)"
    (concat-dataset (DataSet. [1 1] [3 3]) (DataSet. [-1 -1] [0 0]))
    => #clj_elm.data.DataSet{:classes [1 1 -1 -1], :features [3 3 0 0]}))

(facts "test-shuffle-dataset"
  (facts "(shuffle-dataset dataset)"
    (-> (shuffle-dataset (DataSet. [1 2 3 4 5] [1 2 3 4 5]))
        (#(= (:classes %) (:features %))))
    => true
    (-> (shuffle-dataset (DataSet. [1 2 3 4 5] [1 2 3 4 5]))
        (#(= (:classes %) (:features %))))
    => true
    (-> (shuffle-dataset (DataSet. [1 2 3 4 5] [1 2 3 4 5]))
        (#(= (:classes %) (:features %))))
    => true))


(facts "test-ith-feature-list"
  (facts "(ith-feature-list dataset i)"
    (take 3 (ith-feature-list australian 1))
    => [22.08 22.67 29.58]
    (drop 687 (ith-feature-list australian 1))
    => [18.83 27.42 41]
    (take 3 (ith-feature-list australian 13))
    => [1213 1 1]
    (drop 687 (ith-feature-list australian 14))
    => [1 1 1]
    (ith-feature-list A 1)
    => [1 1 1]))

(facts "test-each-ith-sd"
  (facts "(each-ith-sd dataset)"
    (take 3 (each-ith-sd (:features (data-set australian 14))))
    => [0.46748239187205504 11.853272772971627 4.978163248528541]))

(facts "test-each-ith-mean"
  (facts "(each-ith-mean dataset)"
    (take 3 (each-ith-mean (:features (data-set australian 14))))
    => [0.6782608695652174 31.56820289855064 4.758724637681158]
    (count (each-ith-mean (:features (data-set australian 14))))
    => 14
    (each-ith-mean A)
    => [1.0 1.0 1.0 1.0]
    (each-ith-mean B)
    => [5.0 30.0 0.0]))

(facts "test-normalize"
  (let [features (:features (data-set australian 14))]
    (facts "(normalize dataset)"
      (->> (range 0 (num-of-feature features))
           (map #(ith-feature-list (normalize features) %))
           (map st/mean)
           (map #(Math/round %)))
      => (repeat (num-of-feature features) 0))
    (->> (range 0 (num-of-feature features))
         (map #(ith-feature-list (normalize features) %))
         (map st/sd)
         (map #(Math/round %)))
    => (repeat (num-of-feature features) 1)))

(facts "test-remove-list"
  (facts "(remove-list lst a b)"
    (remove-list [0 1 2 3 4] 1 2)
    => [0 3 4]
    (remove-list [[0 0 0 0] [1 1 1 1] [2 2 2 2] [3 3 3 3] [4 4 4 4] [5 5 5 5]]
                 3 4)
    => [[0 0 0 0] [1 1 1 1] [2 2 2 2] [5 5 5 5]]))

(facts "test-extract-list"
  (facts "(extract-list lst a b)"
    (extract-list [0 1 2 3 4] 1 2)
    => [1 2]
    (extract-list [[0 0 0 0] [1 1 1 1] [2 2 2 2] [3 3 3 3] [4 4 4 4] [5 5 5 5]]
                  3 4)
    => [[3 3 3 3] [4 4 4 4]]))
