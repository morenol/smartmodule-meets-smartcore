use fluvio_smartmodule::{smartmodule, Record, RecordData, Result};
use smartcore::linalg::basic::{arrays::Array2, matrix::DenseMatrix};
use sms_data_clean::bag_of_words;

mod model {
    include!(concat!(env!("OUT_DIR"), "/model.rs"));
}
#[smartmodule(map)]
pub fn map(record: &Record) -> Result<(Option<RecordData>, RecordData)> {
    let key = record.key.clone();
    let sms = std::str::from_utf8(record.value.as_ref())?;

    let tokens = std::str::from_utf8(record.value.as_ref())?
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>()
        .split_ascii_whitespace()
        .map(ToString::to_string)
        .collect::<Vec<String>>();

    let x = bag_of_words::<usize>(tokens, &model::vocabulary());
    let x = DenseMatrix::from_row(&x);
    let model = model::naive_bayes_model();
    let y = model.predict(&x)?;
    let spam = y[0] == 1;

    let value = serde_json::json!({
        "sms": sms,
        "spam": spam,
    })
    .to_string();

    Ok((key, value.into()))
}
