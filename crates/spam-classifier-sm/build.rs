use smartcore::metrics::accuracy::Accuracy;
use smartcore::metrics::Metrics;
use smartcore::model_selection::train_test_split;
use smartcore::naive_bayes::multinomial::MultinomialNB;
use sms_data_clean::create_smartcore_input;

use std::env;
use std::path::Path;
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../SMSSPamCollection");

    let out_dir = env::var("OUT_DIR").unwrap();
    let (x, y, vocabulary) =
        create_smartcore_input::<usize, _>("../../SMSSPamCollection").expect("failed to init");

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.7, false, Some(10));

    let model = MultinomialNB::fit(&x_train, &y_train, Default::default()).expect("failed to fit");
    let y_result = model.predict(&x_test).expect("failed to predict");
    let accuracy = Accuracy::new().get_score(&y_test, &y_result);
    assert!(accuracy > 0.9);
    let model_string = format!(
        "
        use smartcore::linalg::basic::matrix::DenseMatrix;
        use smartcore::naive_bayes::multinomial::MultinomialNB;
        pub fn naive_bayes_model() -> MultinomialNB<usize, usize, DenseMatrix<usize>, Vec<usize>> {{
           serde_json::from_str(r#\"{}\"#).unwrap()
        }}

        pub fn vocabulary() -> ::std::collections::HashMap<String, usize> {{
            serde_json::from_str(r#\"{}\"#).unwrap()
        }}
    ",
        serde_json::to_string(&model).expect("Failed to serialize model"),
        serde_json::to_string(&vocabulary).expect("Failed to serialize vocabulary")
    );
    let dest_path = Path::new(&out_dir).join("model.rs");
    std::fs::write(&dest_path, model_string).expect("Failed to generate code");
}
