use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::{collections::HashSet, io::BufRead, path::Path, str::FromStr};

use smartcore::linalg::basic::arrays::{Array1, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::numbers::basenum::Number;
use stopwords::{Language, Stopwords, NLTK};

#[derive(Debug)]
pub enum Label {
    Ham,
    Spam,
}

impl FromStr for Label {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ham" => Ok(Label::Ham),
            "spam" => Ok(Label::Spam),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
pub struct RawData {
    pub label: Label,
    pub sms: String,
}

#[derive(Debug, Default, Clone)]
pub struct TokenizedData {
    pub tokens: Vec<String>,
}

impl RawData {
    pub fn lowercase(self) -> Self {
        Self {
            label: self.label,
            sms: self.sms.to_lowercase(),
        }
    }

    pub fn without_punctuaction(self) -> Self {
        Self {
            label: self.label,
            sms: self
                .sms
                .chars()
                .filter(|c| !c.is_ascii_punctuation())
                .collect(),
        }
    }
}

#[derive(Debug)]
pub struct RawDataset {
    pub data: Vec<RawData>,
}

impl RawDataset {
    pub fn from_file<P>(path: P) -> Result<Self, std::io::Error>
    where
        P: AsRef<Path>,
    {
        let file_data = std::fs::read(path)?;

        let data = file_data
            .lines()
            .map(|line| {
                let line = line?;
                let (label, sms) = line.split_once('\t').ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::Other, "Missing delimeter")
                })?;
                let label = Label::from_str(label)
                    .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid label"))?;
                let sms = sms.to_string();
                Ok(RawData { label, sms })
            })
            .collect::<Result<Vec<_>, std::io::Error>>()?;
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn lowercase(self) -> Self {
        Self {
            data: self.data.into_iter().map(|row| row.lowercase()).collect(),
        }
    }

    pub fn without_punctuaction(self) -> Self {
        Self {
            data: self
                .data
                .into_iter()
                .map(|row| row.without_punctuaction())
                .collect(),
        }
    }

    pub fn tokenize(self) -> Dataset {
        let (labels, data) = self
            .data
            .into_iter()
            .map(|row| {
                (
                    row.label,
                    TokenizedData {
                        tokens: row.sms.split_whitespace().map(|s| s.to_string()).collect(),
                    },
                )
            })
            .unzip();
        Dataset { labels, data }
    }
}

#[derive(Debug)]
pub struct Dataset {
    pub labels: Vec<Label>,
    pub data: Vec<TokenizedData>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stop_words(self) -> Self {
        let stops: HashSet<_> = NLTK::stopwords(Language::English).unwrap().iter().collect();
        Self {
            labels: self.labels,
            data: self
                .data
                .into_iter()
                .map(|row| TokenizedData {
                    tokens: row
                        .tokens
                        .into_iter()
                        .filter(|p| !stops.contains(&p.as_str()))
                        .collect(),
                })
                .collect(),
        }
    }

    pub fn to_smartcore<T: Number>(
        self,
    ) -> Result<(DenseMatrix<T>, Vec<T>, HashMap<String, usize>), std::io::Error> {
        let labels = self
            .labels
            .into_iter()
            .map(|label| match label {
                Label::Spam => T::one(),
                Label::Ham => T::zero(),
            })
            .collect::<Vec<_>>();
        let mut vocabulary = HashMap::new();
        let mut index = 0;

        for word in self.data.clone().into_iter().flat_map(|data| data.tokens) {
            if let Entry::Vacant(entry) = vocabulary.entry(word) {
                entry.insert(index);
                index += 1;
            }
        }

        let data = self
            .data
            .into_iter()
            .map(|data| bag_of_words::<T>(data.tokens, &vocabulary))
            .collect::<Vec<_>>();

        let data_m = DenseMatrix::from_2d_vec(&data);

        Ok((data_m, labels, vocabulary))
    }
}

pub fn create_smartcore_input<T: Number, P: AsRef<Path>>(
    path: P,
) -> Result<(DenseMatrix<T>, Vec<T>, HashMap<String, usize>), std::io::Error> {
    RawDataset::from_file(path)
        .expect("creation failed")
        .lowercase()
        .without_punctuaction()
        .tokenize()
        .stop_words()
        .to_smartcore()
}

pub fn bag_of_words<T: Number>(tokens: Vec<String>, vocabulary: &HashMap<String, usize>) -> Vec<T> {
    let mut m = Vec::zeros(vocabulary.len());

    for token in tokens {
        if let Some(index) = vocabulary.get(&token) {
            m.add_element_mut(*index, T::one());
        }
    }

    m
}

#[cfg(test)]
mod test {

    use crate::RawDataset;

    #[test]
    fn test_file_load() {
        let dataset = RawDataset::from_file("../../SMSSpamCollection").expect("creation failed");
        assert_eq!(dataset.len(), 5574);
        assert_eq!(dataset.data[0].sms.split_ascii_whitespace().count(), 20);
        assert_eq!(dataset.data[1].sms.split_ascii_whitespace().count(), 6);
    }

    #[test]
    fn test_preprocessing() {
        let dataset = RawDataset::from_file("../../SMSSpamCollection")
            .expect("creation failed")
            .lowercase()
            .without_punctuaction()
            .tokenize()
            .stop_words();
        assert_eq!(dataset.len(), 5574);
        assert_eq!(dataset.data[0].tokens.len(), 16);
        assert_eq!(dataset.data[1].tokens.len(), 6);
    }

    #[test]
    fn test_to_smartcore() {
        let (matrix, labels, vocab) = RawDataset::from_file("../../SMSSpamCollection")
            .expect("creation failed")
            .lowercase()
            .without_punctuaction()
            .tokenize()
            .stop_words()
            .to_smartcore::<f64>()
            .expect("Failed to convert to smartcore");
    }
}
