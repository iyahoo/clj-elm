(ns clj-elm.data-test
  (:require [clj-elm.data :refer :all]
            ;; [clojure.core.typed :as t :only [atom doseq let fn defn ref dotimes defprotocol loop for]]
            ;; [clojure.core.typed :refer :all :exclude [atom doseq let fn defn ref dotimes defprotocol loop for]]
            [midje.sweet :refer :all]
            [midje.repl :refer (autotest load-facts)]
            [incanter.core :as c :exclude [update]]
            [incanter.io :as io]
            [incanter.stats :as st]))

(def australians
  (io/read-dataset "data/australian.csv" :delim \, :header true))

(def A (c/to-dataset [[1 1 1 1] [1 1 1 1] [1 1 1 1]]))
(def B (c/to-dataset [[10 20 0] [0 40 0] [20 50 0] [-10 10 1]]))

(facts "test-num-of-feature"
  (fact "(num-of-feature dataset)"
    (num-of-feature australians) => 14
    (num-of-feature A) => 3))

(facts "test-num-of-data"
  (fact "(num-of-data dataset)"
    (num-of-data australians) => 690
    (num-of-data A) => 3))

(facts "test-class-label"
  (fact "(class-label dataset)"
    (take 20 (class-label australians)) 
    => [-1 -1 -1 1 1 1 -1 1 -1 -1 1 1 -1 1 -1 1 1 1 -1 1]))

(facts "test-get-features"
  (fact "(get-features line)"
    (take 2 (map get-features (c/to-vect australians)))
    => [[1 22.08 11.46 2 4 4 1.585 0 0 0 1 2 100 1213]
        [0 22.67     7 2 8 4 0.165 0 0 0 0 2 160    1]]
    (get-features (last (c/to-vect australians)))
    => [1 41 0.04 2 10 4 0.04 0 1 1 0 1 560 1]
    (get-features (first (c/to-vect A))) => [1 1 1]))

(facts "test-ith-feature-list"
  (fact "(ith-feature-list dataset i)"
    (take 3 (ith-feature-list australians 1))
    => [22.08 22.67 29.58]
    (drop 687 (ith-feature-list australians 1))
    => [18.83 27.42 41]
    (take 3 (ith-feature-list australians 13))
    => [1213 1 1]
    (drop 687 (ith-feature-list australians 14))
    => [1 1 1]
    (ith-feature-list A 1)
    => [1 1 1]))

(facts "test-each-ith-mean"
  (fact "(each-ith-mean dataset)"
    (take 3 (each-ith-mean australians))
    => [0.6782608695652174 31.56820289855064 4.758724637681158]
    (count (each-ith-mean australians))
    => 15
    (each-ith-mean A)
    => [1.0 1.0 1.0 0]
    (each-ith-mean B)
    => [5.0 30.0 0]))

(facts "test-each-ith-sd"
  (fact "(each-ith-sd dataset)"
    (take 3 (each-ith-sd australians))
    => [0.46748239187205504 11.853272772971627 4.978163248528541]))

(facts "test-normalize"
  (fact "(normalize dataset)"
    (->> (range 1 (inc (num-of-feature australians)))
         (map #(ith-feature-list (normalize australians) %))
         (map st/mean)
         (map #(Math/round %)))
    => (repeat (num-of-feature australians) 0)
    (->> (range 1 (inc (num-of-feature australians)))
         (map #(ith-feature-list (normalize australians) %))
         (map st/sd)
         (map #(Math/round %)))
    => (repeat (num-of-feature australians) 1)))

