from pathlib import Path
from nlp_tasks.language_detection.ld import detect_language_formforfile
from nlp_tasks.named_entity_recognition.ner import named_entity_recognition
from nlp_tasks.summarization.sum import summarize
from nlp_tasks.text2image.t2i import generate_image
# from nlp_tasks.text_similarity.ts import similarity_matrix
from nlp_tasks.translation.tl import translate
from nlp_tasks.zero_shot_classification.zsc import zs_class
from preprocessing.clean_text import clean_text
import os.path
import shutil
import os
import pandas
from nlp_tasks.summarization.sum import summarize
from nlp_tasks.named_entity_recognition.ner import named_entity_recognition
from nlp_tasks.translation.tl import translate
from nlp_tasks.zero_shot_classification.zsc import zs_class
from nlp_tasks.text2image.t2i import generate_image, generate_filename_png_from_text
import glob
from config import in_path_toplevel, out_path_toplevel, tag_candidates_path


def run_if_outtext_not_existant(f, *args, outfile):
    if not os.path.exists(outfile):
        # print("File ", outfile, " does not exist yet.")
        outlistlines = f(*args)
        with open(outfile, "a") as file:
            file.writelines(outlistlines)
    else:
        # print("File ", outfile, " exists.")
        pass
    with open(outfile) as f:
        lines = f.read().splitlines()
    return lines


class txtdecoration:
    def __init__(
        self, in_folder, in_file, out_folder, tag_candidates_path=None
    ) -> None:

        self.in_filename = in_file
        self.in_filename_noextension = self.in_filename.split(".")[0]
        self.in_filepath_absolute = os.path.join(in_folder, in_file)

        with open(self.in_filepath_absolute) as f:
            lines = f.readlines()
        raw_text = " ".join(lines)
        self.in_file_content = clean_text(raw_text)
        self.out_folder_top = out_folder
        self.out_folder_named = os.path.join(
            self.out_folder_top, self.in_filename_noextension
        )

        if not os.path.exists(self.out_folder_named):
            os.makedirs(self.out_folder_named)

        self.tag_candidates_path = tag_candidates_path

    def _cp_original_file(self, outfile):
        shutil.copy(self.in_filepath_absolute, outfile)

    def cp_original_file(self):
        outfile = os.path.join(self.out_folder_named, self.in_filename)
        if not os.path.exists(outfile):
            self._cp_original_file(outfile)

    def nlpt_language_detection(self):
        outfile = os.path.join(self.out_folder_named, "language_detection.txt")
        lines = run_if_outtext_not_existant(
            detect_language_formforfile, self.in_file_content, outfile=outfile
        )

    def nlpt_namedentityrecognition(self):
        outfile_locs = os.path.join(
            self.out_folder_named, "namedentityrecognition_locations.csv"
        )
        outfile_pers = os.path.join(
            self.out_folder_named, "namedentityrecognition_persons.csv"
        )
        outfile_orgs = os.path.join(
            self.out_folder_named, "namedentityrecognition_organizations.csv"
        )
        if (
            not os.path.exists(outfile_locs)
            or not os.path.exists(outfile_pers)
            or not os.path.exists(outfile_orgs)
        ):
            locs, pers, orgs = named_entity_recognition(self.in_file_content)
            print("NER for ", self.in_filepath_absolute)

            f = open(outfile_locs, 'w+')
            locs.to_csv(f, index = False, header = True, sep = ',', encoding = 'utf-8')
            f.close()

            f = open(outfile_pers, 'w+')
            pers.to_csv(f, index = False, header = True, sep = ',', encoding = 'utf-8')
            f.close()

            f = open(outfile_orgs, 'w+')
            orgs.to_csv(f, index = False, header = True, sep = ',', encoding = 'utf-8')
            f.close()
            
            # locs.to_csv(outfile_locs, sep=",", index=False, mode="w+")
            # pers.to_csv(outfile_pers, sep=",", index=False, mode="w+")
            # orgs.to_csv(outfile_orgs, sep=",", index=False, mode="w+")

    def nlpt_summarization(self):
        outfile = os.path.join(self.out_folder_named, "summary.txt")
        s = run_if_outtext_not_existant(
            summarize, self.in_file_content, outfile=outfile
        )

    # def nlpt_textsimilarity(self):
    #     pass

    def nlpt_summarization_en(self):
        outfile = os.path.join(self.out_folder_named, "summary_en.txt")
        summary_path = os.path.join(self.out_folder_named, "summary.txt")
        if not os.path.exists(outfile):
            with open(summary_path) as f:
                lines = f.readlines()
            summary = " ".join(lines)
            summary = summary.replace("\n", "")
            run_if_outtext_not_existant(translate, "de", "en", summary, outfile=outfile)

    def nlpt_text2image(self):

        summary_en_path = os.path.join(self.out_folder_named, "summary_en.txt")
        
        if os.path.exists(summary_en_path):
            
            with open(summary_en_path) as f:
                lines = f.readlines()
            summary_en = " ".join(lines)
            summary_en = summary_en.replace("\n", "")
            
            img_filename = generate_filename_png_from_text(summary_en)
            img_path = os.path.join(self.out_folder_named, img_filename)
            
            if not os.path.exists(img_path):
                generate_image(summary_en, img_path)

    def nlpt_translation(self):
        outfile = os.path.join(self.out_folder_named, "translation_fr.txt")
        run_if_outtext_not_existant(
            translate, "de", "fr", self.in_file_content, outfile=outfile
        )

    def _tag_candidates_list(self):
        if self.tag_candidates_path is not None:
            with open(self.tag_candidates_path) as f:
                content_list = f.readlines()
                content_list = [s.replace("\n", "") for s in content_list]
                content_list = list(set(content_list))
            return content_list
        else:
            return None

    def nlpt_zeroshotclassification(self):
        outfile_z = os.path.join(self.out_folder_named, "zs_classification.csv")
        if not os.path.exists(outfile_z):
            print("Working on: ", outfile_z)
            tag_candidates = self._tag_candidates_list()
            df = zs_class(
                candidate_labels=tag_candidates,
                sequence_to_classify=(self.in_file_content),
            )
            f = open(outfile_z, 'w+')
            df.to_csv(f, index = False, header = True, sep = ',', encoding = 'utf-8')
            f.close()

    def process_all(self):
        try:
            self.cp_original_file()
        except Exception as e:
            print("----")
            print("CP FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)
        try:
            self.nlpt_language_detection()
        except Exception as e:
            print("----")
            print("LD FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)
        try:
            self.nlpt_namedentityrecognition()
        except Exception as e:
            print("----")
            print("NER FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)
        try:
            self.nlpt_summarization()
        except Exception as e:
            print("----")
            print("SUM FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)
        try:
            self.nlpt_summarization_en()
        except Exception as e:
            print("----")
            print("SUME FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)
        try:
            self.nlpt_translation()
        except Exception as e:
            print("----")
            print("TL  FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)
        try:
            self.nlpt_zeroshotclassification()
        except Exception as e:
            print("----")
            print("ZSC FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)
        try:
            self.nlpt_text2image()
        except Exception as e:
            print("----")
            print("T2I FAILED FOR: ", self.out_folder_named)
            print("The error raised is: ", e)


def clean_subfolder_path(subfolder):
    if len(subfolder) > 0:
        if subfolder[0] == "/":
            subfolder = subfolder[1:]
    return subfolder

def outfolder_from_infolder(in_folder_abs, in_path_toplevel, out_path_toplevel):
    subfolder = in_folder_abs.replace(in_path_toplevel, "")
    subfolder = clean_subfolder_path(subfolder)
    out_folder_abs = os.path.join(out_path_toplevel, subfolder)
    return out_folder_abs



if __name__ == "__main__":

    for folder_abs, _, files in os.walk(in_path_toplevel):
        out_folder_abs = outfolder_from_infolder(
            folder_abs, in_path_toplevel, out_path_toplevel
        )
        for f in files:
            print(f)
            td = txtdecoration(folder_abs, f, out_folder_abs, tag_candidates_path)
            td.process_all()

    # for folder_abs, _, files in os.walk(in_path_toplevel):
    #     out_folder_abs = outfolder_from_infolder(
    #         folder_abs, in_path_toplevel, out_path_toplevel
    #     )

    #     d = {}
    #     for f in files:
    #         filepath_abs = os.path.join(folder_abs, f)
    #         filename_noextension = f.split(".")[0]
    #         d[filename_noextension] = filepath_abs

    # from nlp_tasks.text_similarity.ts import similarity_matrix, visualize_similarities

    # import numpy as np
    # v = np.array(list(d.values()))
    # k = np.array(list(d.keys()))

    # file_nr = np.array([int(s.split("_")[0]) for s in k])
    # sort_idx = np.argsort(file_nr)

    # v = v[sort_idx]
    # k = k[sort_idx]
    # k = [h[:20] for h in k]

    # s = similarity_matrix(v, v)
    # visualize_similarities(s, k, k)
    # print()