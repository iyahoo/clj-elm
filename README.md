# clj-elm
[![Build Status](https://secure.travis-ci.org/iyahoo/clj-elm.png)](http://travis-ci.org/iyahoo/clj-elm)
[![Coverage Status](https://coveralls.io/repos/iyahoo/clj-elm/badge.svg?branch=master&service=github)](https://coveralls.io/github/iyahoo/clj-elm?branch=master)

This is Extreme Learning Machine on Clojure. For only 2 class classification.

## Usage

```shell
$ git clone https://github.com/iyahoo/clj-elm.git
$ cd clj-elm
$ lein repl
```

```clojure
(load-file "src/clj_elm/core.clj")
(in-ns 'clj-elm.core)

(def dataset (read-dataset "data/australian.csv" 14 :header true))
;=> #'clj-elm.core/dataset

(def model (train-model dataset 20 :norm true))
;=> #'clj-elm.core/model

(predict model (first (normalize (:features dataset))))
;=> -1
```

You can use dataset CSV or lib-svm form.

```clojure
;; csv
(read-dataset path-to-csv-dataset class-column :header true-or-false)
;; lib-svm
(read-dataset path-to-lib-svm-form-dataset)
```

Cross validate test:

```clojure
(cross-validate dataset 100 10)
```

This will print out L, length each data, TP, FP, TN, FN, Accuracy, Recall and Precision.

Command line

```shell
lein run "path/to/negative.csv" location-class header-true-or-false <and same 3 args for positive> number-of-hidden-neuron number-of-cross-validation sign-reverse-flag
```

## License

Copyright © 2015 iyhoo.

Distributed under the Eclipse Public License.
