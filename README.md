# clj-elm

This is Extreme Learning Machine on Clojure. For only 2 class classification.  

## Usage

`(def dataset (data/read-dataset "data/australian.csv" 14 :header true))`
`;=> #'clj-elm.core/dataset`

`(def model (train-model dataset 20 :norm true))`
`;=> #'clj-elm.core/model`

`(predict model (first (data/normalize (:features dataset))))`
`;=> -1`
Cross validate test:

`(cross-validate dataset 10 100)`

## License

Copyright © 2015 iyhoo.

Distributed under the Eclipse Public License.
