import csv
import heapq
import json
from enum import Enum
from typing import List, Dict, Iterator, Optional, Callable, Union
import pickle

import numpy as np
import requests
from io import StringIO

from collections import defaultdict


class DB:
    """
    Contains the preprocessed data from the TCGA
    """

    def __init__(self, cases: Dict[str, 'Case'], samples: Dict[str, 'Sample']):
        self.cases = cases
        self.samples = samples  # Dict[str, 'Sample']

    def get_sample(self, sample_id: str) -> 'Sample':
        return self.samples[sample_id]

    def get_samples(self) -> Iterator['Sample']:
        return self.samples.values()

    def get_case(self, case_id: str) -> 'Case':
        return self.cases[case_id]

    def get_cases(self) -> Iterator['Case']:
        return self.cases.values()

    def add_case(self, case: 'Case') -> bool:
        """
        Adds a case if a case with the same id isn't in the database
        :return: case added?
        """
        if case.id in self.cases:
            return False
        self.cases[case.id] = case
        #print("Added case {}".format(case.id))
        return True

    def add_sample(self, sample: 'Sample') -> bool:
        """
        Adds a sample if a sample with the same id isn't in the database
        :return: sample added?
        """
        if sample in self.samples:
            print("Skip sample {}".format(sample.id))
            return False
        self.samples[sample.id] = sample
        return True

    def clear(self):
        self.cases.clear()
        self.samples.clear()

    def serialize(self) -> dict:
        return {"MI_RNA_NAMES": MI_RNA_NAMES,
                "samples": {id:sample.serialize() for (id, sample) in self.samples.items()},
                "cases": {id:case.serialize() for (id, case) in self.cases.items()}}

    @staticmethod
    def deserialize(data: dict) -> 'DB':
        db = DB({}, {})
        global MI_RNA_NAMES
        MI_RNA_NAMES = data["MI_RNA_NAMES"]
        for sample in data["samples"].values():
            assert db.add_sample(Sample.deserialize(sample, db))
        for case in data["cases"].values():
            assert db.add_case(Case.deserialize(case, db))
        return db

    def store(self, filename: str):
        with open(filename, "w") as f:
            serialized = self.serialize()
            json.dump(serialized, f)

    @staticmethod
    def load(filename: str) -> 'DB':
        with open(filename, "r") as f:
            return DB.deserialize(json.load(f))

    @staticmethod
    def check(filename: str):
        db = DB.load(filename)

    def pull_all(self):
        r = requests.get('https://api.gdc.cancer.gov/cases')
        json = r.json()
        if json["warnings"]:
            print(json["warnings"])
        data = json["data"]
        pages = data["pagination"]["pages"]
        for i in range(pages):
            self._pull_page(i * 10)

    def _pull_page(self, page: int):
        r = requests.get('https://api.gdc.cancer.gov/cases', params={"from": page, "expand": ["files", "samples"]})
        json = r.json()
        if json["warnings"]:
            print(json["warnings"])
        data = json["data"]
        for d in data["hits"]:
            self.pull_case(d["case_id"], d)

    def pull_all_cases_from_dict_list(self, d: List[dict]):
        for entry in d:
            self.pull_case(entry["case_id"])

    def _pull_data(self, sample_id: str, file_id: str) -> 'MiRNAProfile':
        r = requests.get('https://api.gdc.cancer.gov/data/' + file_id)
        row_titles = []
        reads = []
        for row in csv.reader(StringIO(r.text), delimiter="\t"):
            if row[0].startswith("hsa"):
                row_titles.append(row[0])
                reads.append(float(row[2]))
        return MiRNAProfile(sample_id, reads, row_titles)

    def _pull_sample_information(self, case_id: str, sample_id: str):
        r = requests.get('https://api.gdc.cancer.gov/v0/files/' + sample_id, params={"expand": "metadata_files", "fields": "cases.project.disease_type,state,md5sum,access,data_format,data_type,data_category,file_name,file_size,file_id,platform,experimental_strategy,center.short_name,cases.case_id,cases.project.project_id,cases.samples.sample_type,cases.samples.portions.portion_id,cases.samples.portions.analytes.analyte_id,cases.samples.portions.analytes.aliquots.aliquot_id,annotations.annotation_id,annotations.entity_id,tags,submitter_id,archive.archive_id,archive.submitter_id,archive.revision,associated_entities.entity_id,associated_entities.entity_type,associated_entities.case_id,analysis.analysis_id,analysis.workflow_type,analysis.updated_datetime,analysis.input_files.file_id,analysis.metadata.read_groups.read_group_id,analysis.metadata.read_groups.is_paired_end,analysis.metadata.read_groups.read_length,analysis.metadata.read_groups.library_name,analysis.metadata.read_groups.sequencing_center,analysis.metadata.read_groups.sequencing_date,downstream_analyses.output_files.access,downstream_analyses.output_files.file_id,downstream_analyses.output_files.file_name,downstream_analyses.output_files.data_category,downstream_analyses.output_files.data_type,downstream_analyses.output_files.data_format,downstream_analyses.workflow_type,downstream_analyses.output_files.file_size,index_files.file_id"})
        json = r.json()
        if json["warnings"]:
            print(json["warnings"])
        misc = json["data"]
        data = json["data"]
        #assert data["data_type"] == "miRNA Expression Quantification"
        assert len(data["cases"][0]["samples"]) == 1
        sample = data["cases"][0]["samples"][0]
        sample_type = sample["sample_type"].lower()
        type = data["cases"][0]["project"]["disease_type"]
        if "normal" in sample_type and "blood" not in sample_type:
            type = NORMAL_TISSUE
        mi_rna_profile = self._pull_data(sample_id, sample_id)
        return Sample(sample_id, self, case_id, type, "blood" in sample["sample_type"], mi_rna_profile, misc)

    def pull_all_cases(self, case_ids: List[str]):
        for case_id in case_ids:
            self.pull_case(case_id)

    def pull_case(self, case_id: str, data: dict = None):
        if case_id in self.cases:
            return
        print("Pull case {}".format(case_id))
        if not data:
            r = requests.get('https://api.gdc.cancer.gov/cases/' + case_id, params={"expand": ["files", "samples"]})
            json = r.json()
            if json["warnings"]:
                print(json["warnings"])
            data = json["data"]
        misc = data
        sample_ids = []
        has_normal_or_blood = False
        has_tumour = False
        for file_d in data["files"]:
            if file_d["data_type"] == "miRNA Expression Quantification":
                sample_id = file_d["file_id"]
                print("Pulled sample " + sample_id)
                sample = self._pull_sample_information(case_id, sample_id)
                self.add_sample(sample)
                sample_ids.append(sample_id)
                if sample.is_blood or sample.tissue_type == NORMAL_TISSUE:
                    has_normal_or_blood = True
                else:
                    has_tumour = True
        if len(sample_ids) == 0:
            print("No miRNA expression samples for case {}".format(case_id))
        else:
            self.add_case(Case(case_id, self, sample_ids, misc))
            print("→ {} cases and {} samples ({} normal samples) in db "
                  .format(len(self.cases), len(self.samples), self.normal_sample_count()))

    def sample_arrs_and_features(self, blood_normals: bool = True,
                                     tumour: bool = True,
                                     min_samples: int = 1) -> (np.array, np.array):
        samples = []
        classification = []
        samples_per_tumours = self.samples_per_tumour(blood_normals, tumour)
        for sample in self.filtered_samples(blood_normals, tumour):
            if samples_per_tumours[sample.tissue_type] >= min_samples:
                samples.append(np.array(sample.mi_rna_profile.reads_per_million))
                classification.append(sample.tissue_type)
        return np.array(samples), classification

    def samples_per_tumour(self, blood_normals: bool = True, tumour: bool = True) -> Dict[str, int]:
        ret = {}
        for sample in self.filtered_samples(blood_normals, tumour):
            t = sample.tissue_type
            if t not in ret:
                ret[t] = 1
            else:
                ret[t] += 1
        return ret

    def filtered_samples(self, blood_normals: bool, tumour: bool) -> Iterator['Sample']:
        for sample in self.samples.values():
            if (sample.is_blood and blood_normals) or (not sample.is_blood and not sample.tissue_type == NORMAL_TISSUE and tumour):
                yield sample

    def normal_sample_count(self) -> int:
        return len([sample for sample in self.samples.values() if sample.tissue_type == NORMAL_TISSUE])

    def samples_per_type(self) -> Dict[str, List['Sample']]:
        d = defaultdict(lambda: [])
        for sample in self.samples.values():
            d[sample.tissue_type].append(sample)
        return d

    def remove_mirnas(self, mirnas: List[Union[str,int]]):
        global MI_RNA_NAMES
        mirnas = [x if type(x) is int else MI_RNA_NAMES.index(x) for x in mirnas]
        names = []
        for i, name in enumerate(MI_RNA_NAMES):
            if i not in mirnas:
                names.append(name)
        MI_RNA_NAMES = names
        mirnas = set(mirnas)
        for sample in self.samples.values():
            profile = sample.mi_rna_profile
            profile.mi_rna_names = MI_RNA_NAMES
            new_vals = []
            for i, val in enumerate(profile.reads_per_million):
                if i in mirnas:
                    new_vals.append(val)
            profile.reads_per_million = np.array(new_vals)

    def remove_blood_normals(self):
        for case in self.cases.values():
            case.filter_samples(lambda sample: not sample.is_blood, remove_from_db=True)

    def remove_samples(self, type: str):
        """ Remove all samples with the given type """
        for case in self.cases.values():
            case.filter_samples(lambda sample: sample.tissue_type != type, remove_from_db=True)

    def discard_tumours(self, min_tumour_samples: int):
        for tumour, count in self.samples_per_tumour():
            if count < min_tumour_samples:
                self.remove_samples(tumour)

    def samples_per_mirna(self) -> Dict[str, int]:
        """ Count for each miRNA the samples that have a non zero value for the miRNA """
        ret = []
        for sample in self.samples.values():
            if not ret:
                ret = [0 for k in sample.mi_rna_profile.mi_rna_names]
            for i, val in enumerate(sample.mi_rna_profile.reads_per_million):
                if val > 0:
                    ret[i] += 1
        return {MI_RNA_NAMES[i]:val for i, val in enumerate(ret)}

    def discard_mirnas(self, min_sample_perc: int):
        """ Remove miRNAs that have missing values for more than n% of samples """
        min_samples_to_discard = np.math.ceil((1 - min_sample_perc / 100) * len(self.samples))
        discarded_mirnas = [miRNA for miRNA, count in self.samples_per_mirna().items() if count < min_samples_to_discard]
        self.remove_mirnas(discarded_mirnas)

    def print_stats(self, blood: bool, only_blood: bool, min_samples: int):
        """ Show some stats about the samples in the database """
        samples_per_tumour = self.samples_per_tumour(blood, not only_blood)
        used_lines = []
        omitted_lines = []
        for tumour in sorted(samples_per_tumour.keys()):
            line = "    {:50s} {:10d}".format(tumour, samples_per_tumour[tumour])
            if samples_per_tumour[tumour] >= min_samples:
                used_lines.append(line)
            else:
                omitted_lines.append(line)
        if len(used_lines):
            print("--- Samples per tumour (for {} tumours with enough samples)".format(len(used_lines)))
            for line in used_lines:
                print(line)
        if len(omitted_lines):
            print("--- {} Tumours which are omitted, because they have less than {} samples"
                  .format(len(omitted_lines), ctx.obj["min_samples"]))
            for line in omitted_lines:
                print(line)
        print("{} normal samples".format(self.normal_sample_count()))
        print("Overall {} cases and {} samples".format(len(self.cases), len(self.samples)))
        print("{} miRNAs".format(len(MI_RNA_NAMES)))

NORMAL_TISSUE = "normal"


def is_tumour_type(tissue_type: str):
    return tissue_type != NORMAL_TISSUE


def knn_impute(db: DB, k: int = 5):
    known_pairs = [] # type: List[Tuple[Sample, Sample]]
    # [(tumour sample, normal sample)]
    cases_with_missing = []  # type: List[Case]
    for case in db.cases.values():
        if case.has_sample_of_type(NORMAL_TISSUE):
            tumour_sample = case.get_tumour_sample()
            if tumour_sample: # just in case
                # a correct sample
                known_pairs.append((tumour_sample, case.get_sample_of_type(NORMAL_TISSUE)))
        else:
            cases_with_missing.append(case)

    for case in cases_with_missing:
        tumour_sample = case.get_tumour_sample()
        calculated_distances = [(t.distance(tumour_sample), n) for i, (t, n) in enumerate(known_pairs)]
        k_best = heapq.nlargest(k, calculated_distances, key=lambda x: x[0])
        dest_sum = sum(1 / (dist * 1.0) for dist, n in k_best)
        avg_vec = np.average([n.mi_rna_profile.reads_per_million * 1 / dist / dest_sum for dist, n in k_best])
        new_sample_id = case.id + "avg_normal"
        new_sample = Sample(new_sample_id, db, case.id, NORMAL_TISSUE, False, MiRNAProfile(new_sample_id, avg_vec), misc={})
        db.add_sample(new_sample)
        case.sample_ids.append(new_sample_id)

class Case:
    """
    A case with associated samples
    """

    def __init__(self, id: str, db: DB, sample_ids: List[str], misc: dict):
        self.db = db
        self.id = id
        self.sample_ids = sample_ids
        self.misc = misc

    def get_samples(self) -> List['Sample']:
        return [self.db.get_sample(id) for id in self.sample_ids]

    def has_sample_of_type(self, type: str):
        return any(sample.tissue_type == type for sample in self.get_samples())

    def serialize(self) -> dict:
        return {"id": self.id, "sample_ids": self.sample_ids, "misc": self.misc}

    @staticmethod
    def deserialize(data: dict, db: DB) -> 'Case':
        return Case(data["id"], db, data["sample_ids"], data["misc"])

    def get_tumour_sample(self) -> Optional['Sample']:
        for sample in self.get_samples():
            if is_tumour_type(sample.tissue_type):
                return sample
        return None

    def get_sample_of_type(self, type: str) -> Optional['Sample']:
        for sample in self.get_samples():
            if sample.tissue_type == type:
                return sample
        return None

    def filter_samples(self, filter: Callable[['Sample'], bool], remove_from_db: bool):
        rem = []
        for sample_id in self.sample_ids:
            if not filter(self.db.get_sample(sample_id)):
                rem.append(sample_id)
                if remove_from_db:
                    self.db.samples.pop(sample_id)
        for r in rem:
            self.sample_ids.remove(r)


class Sample:
    """
    A tissue sample with associated miRNA expression profile
    """

    def __init__(self, id: str, db: DB, case_id: str, tissue_type: str, is_blood: bool, mi_rna_profile: 'MiRNAProfile', misc: dict):
        self.id = id
        self.db = db
        self.case_id = case_id
        self.tissue_type = tissue_type
        self.is_blood = is_blood
        self.mi_rna_profile = mi_rna_profile
        self.misc = misc

    def get_case(self) -> Case:
        return self.db.get_case(self.case_id)

    def serialize(self) -> dict:
        return {"id": self.id,
                "case_id": self.case_id,
                "tissue_type": self.tissue_type,
                "is_blood": self.is_blood,
                "mi_rna_profile": self.mi_rna_profile.serialize(),
                "misc": self.misc}

    @staticmethod
    def deserialize(data: dict, db: DB) -> 'Sample':
        return Sample(data["id"], db, data["case_id"], data["tissue_type"], data["is_blood"],
                      MiRNAProfile.deserialize(data["mi_rna_profile"]), data["misc"])

    def distance(self, other_sample: 'Sample') -> float:
        return self.mi_rna_profile.distance(other_sample.mi_rna_profile)

MI_RNA_NAMES = None


class MiRNAProfile:
    """
    The miRNA profile for one specific sample
    """

    def __init__(self, sample_id: str, reads_per_million: List[float], mi_rna_names: List[str] = None):
        global MI_RNA_NAMES
        self.sample_id = sample_id
        if not mi_rna_names:
            assert MI_RNA_NAMES
        else:
            if not MI_RNA_NAMES:
                MI_RNA_NAMES = mi_rna_names
            else:
                assert mi_rna_names == MI_RNA_NAMES
        self.reads_per_million = np.array(reads_per_million)  # type: np.array
        self.mi_rna_names = MI_RNA_NAMES

    def serialize(self) -> tuple:
        return (self.sample_id, self.reads_per_million.tolist())

    @staticmethod
    def deserialize(data: tuple) -> 'MiRNAProfile':
        assert MI_RNA_NAMES
        return MiRNAProfile(data[0], data[1])

    def distance(self, other_profile: 'MiRNAProfile') -> float:
        return np.linalg.norm(other_profile.reads_per_million - self.reads_per_million)