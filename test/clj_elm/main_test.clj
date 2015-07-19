(ns clj-elm.main-test
  (:require [clj-elm.main :refer :all]
            [clj-elm.data :as data]
            ;; [clojure.core.typed :as t :only [atom doseq let fn defn ref dotimes defprotocol loop for]]
            ;; [clojure.core.typed :refer :all :exclude [atom doseq let fn defn ref dotimes defprotocol loop for]]
            [midje.sweet :refer :all]
            [midje.repl :refer (autotest load-facts)]
            [incanter.core :as c :exclude [update]]))


