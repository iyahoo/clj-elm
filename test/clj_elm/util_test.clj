(ns clj-elm.util-test
  (:require [clj-elm.util :refer :all]
            [midje.sweet :refer :all]
            [midje.repl :refer (autotest load-facts)]))

(facts "test-reftype?"
  (fact "(reftype? obj)"
    (reftype? 3)
    => false
    (reftype? (atom 1))
    => true
    (reftype? (ref 3))
    => true))
