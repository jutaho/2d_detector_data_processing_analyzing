#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description - Processes raw data from file ".his"-format of
#               flatpanel XRD-0822 and writes DICOM file
# Author -      J. Horn

import datetime
import math
import numpy as np
from scipy import signal
import scipy.ndimage
from tifffile import imwrite


class XRDProcessor:
    def __init__(self, file, med_filt_kernel_size):
        self.file = file
        self.med_filt_kernel_size = med_filt_kernel_size
        self.date = datetime.datetime.now()
        self.date_time = self.date.strftime("%Y-%m-%d %H:%M:%S")
        self.date = self.date.strftime("%Y%m%d")
        self.flag_no_exp = False
        self.pxl_shift_in_mm = 0.2
        self.bit_resolution = 2**16 - 1
        self.violation_limit_adc = self.bit_resolution - 10
        self.tol_amount_of_adc_limit_viol = 25
        self.init_num_of_bg_fr = 5
        self.integration_thres_factor = 6
        self.bad_pxl_factor = 3
        self.bg = 0

     def process_frames(self):
         self.load_correction_matrices()
         self.read_raw_frames()
         self._define_exceeding_pixels()
         self._classify_frames()
         self._background_correction()
         self._bad_pixel_correction()
         self._sensitivity_correction()
         self._calculate_noise_and_signal()
         self._write_log()    

    def load_correction_matrices(self):
        self.pxl_gain_matrix = np.load("pxlSensitivityMatrix.npy")

    def read_raw_frames(self):
        with open(self.file, mode="rb") as header:
            header_info = np.fromfile(header, np.uint16, count=16, offset=0)
            self.num_of_frames = header_info[10]
            self.num_of_pxl_dim = header_info[8:10]

        with open(self.file, mode="rb") as raw_data:
            self.frames = np.fromfile(
                raw_data,
                np.uint16,
                count=int(self.num_of_frames) * int(self.num_of_pxl_dim[0]) * int(self.num_of_pxl_dim[1]),
                offset=100
            ).reshape((self.num_of_frames, self.num_of_pxl_dim[0], self.num_of_pxl_dim[1]))


    def _define_exceeding_pixels(self):
        pxl_viol_limit_adc_per_frame = np.array(self.frames > self.violation_limit_adc)
        self.count_pxl_over_viol_limit = [
            np.sum(pxl_viol_limit_adc_per_frame[frames])
            for frames in range(len(self.frames))
        ]

    def _classify_frames(self):
        med_sig_vector = np.median(self.frames, axis=(1, 2))
        std_sig_vector = np.std(self.frames, axis=(1, 2))

        index_std_sig_eq_0 = np.concatenate(np.where(med_sig_vector + std_sig_vector == 0))
        self.last_frame_index = self.num_of_frames if not np.any(index_std_sig_eq_0) else index_std_sig_eq_0[0]

        print(f"Total number of frames: {self.num_of_frames}")
        print(f"Total number of valid frames: {self.last_frame_index}")

        init_bg_img = np.mean(self.frames[0:self.init_num_of_bg_fr], axis=0)
        init_bg_corr_img = self.frames - init_bg_img

        init_exp_thres = (
            np.mean(init_bg_corr_img[0:self.init_num_of_bg_fr])
            + 6 * np.std(init_bg_corr_img[0:self.init_num_of_bg_fr])
        )

        exposure_arrays = init_bg_corr_img > init_exp_thres
        exp_count_per_fr = np.sum(exposure_arrays, axis=(1, 2))

        self.fr_indices_beam_on = np.concatenate(np.where(exp_count_per_fr > 1000))

        if np.sum(self.fr_indices_beam_on) == 0:
            self.flag_no_exp = True
            print("No exposed frames detected...")
            self.bg = [np.mean(self.frames, axis=0), np.std(self.frames, axis=0)]
            return

        self.num_of_exp_frames = len(self.fr_indices_beam_on)
        self._define_background_images()

    def _define_background_images(self):
        self.fr_indices_bg_pre = range(self.fr_indices_beam_on[0])
        self.num_of_bg_frames_pre = len(self.fr_indices_bg_pre)
        self.fr_indices_bg_post = range(self.fr_indices_beam_on[-1] + 1, self.last_frame_index)
        self.num_of_bg_frames_post = len(self.fr_indices_bg_post)

        if self.num_of_bg_frames_post > 1:
            self.bg_img_post = np.mean(self.frames[self.fr_indices_bg_post[0]:self.fr_indices_bg_post[1]], axis=0)
        elif self.num_of_bg_frames_post == 1:
            self.bg_img_post = self.frames[self.fr_indices_bg_post]
        else:
            self.bg_img_post = 0

        print(f"{self.num_of_bg_frames_pre} frames pre exposure, from index {self.fr_indices_bg_pre[0]} to {self.fr_indices_bg_pre[-1]}")
        print(f"{self.num_of_exp_frames} exposed frames, from index {self.fr_indices_beam_on[0]} to {self.fr_indices_beam_on[-1]}")
        print(f"{self.num_of_bg_frames_post} frames post exposure")

        for index in range(len(self.fr_indices_beam_on) - 1):
            if self.count_pxl_over_viol_limit[self.fr_indices_beam_on[index]] > self.tol_amount_of_adc_limit_viol:
                print("### ADC limit reached ###")

    def _background_correction(self):
        self.bg_img_pre = np.mean(self.frames[self.fr_indices_bg_pre], axis=0)

        self.bg_corr_img_container = self.frames - self.bg_img_pre
        print("Background correction done...")

        temp_mean_of_bg_corr_bg = np.mean(self.bg_corr_img_container[self.fr_indices_bg_pre], axis=0)
        self.temp_noise_per_pxl = np.std(self.bg_corr_img_container[self.fr_indices_bg_pre], axis=0)
        self.bg_corr_bg_summed_up = np.sum(self.bg_corr_img_container[self.fr_indices_bg_pre], axis=0)

        bg_corr_thres = self.bg_corr_img_container > (temp_mean_of_bg_corr_bg + self.integration_thres_factor * self.temp_noise_per_pxl)
        self.bg_corr_img_container = self.bg_corr_img_container * bg_corr_thres
        self.bg_corr_img = np.sum(self.bg_corr_img_container[self.fr_indices_beam_on], axis=0)

    def _bad_pixel_correction(self):
        perm_bad_pxl_matrix_high = self.bg_img_pre > (np.median(self.bg_img_pre) + self.bad_pxl_factor * np.std(self.bg_img_pre))
        perm_bad_pxl_matrix_low = self.bg_img_pre < (np.median(self.bg_img_pre) - self.bad_pxl_factor * np.std(self.bg_img_pre))

        temp_bad_pxl_matrix = 0 if np.sum(self.bg_img_post) == 0 else self.bg_img_post > self.bg_img_pre + np.std(self.bg_img_pre)

        self.bad_pxl_matrix = perm_bad_pxl_matrix_high + perm_bad_pxl_matrix_low + temp_bad_pxl_matrix
        num_bad_pxl = np.sum(self.bad_pxl_matrix) / (int(self.num_of_pxl_dim[0]) * int(self.num_of_pxl_dim[1]))
        print(f"{num_bad_pxl * 100:.1f} % pixel are out of tolerance and will be corrected.")

        rowy, coly = self.bg_corr_img.shape
        corr_img = np.zeros([rowy + 2, coly + 2])
        corr_img[1:rowy + 1, 1:coly + 1] = self.bg_corr_img
        new_matrix = np.zeros([rowy + 2, coly + 2])
        new_matrix[1:rowy + 1, 1:coly + 1] = self.bg_corr_img
        bad_pxl_map = np.zeros([rowy + 2, coly + 2])
        bad_pxl_map[1:rowy + 1, 1:coly + 1] = self.bad_pxl_matrix
        bad_pxl_map = 1 - bad_pxl_map

        for row in range(1, rowy + 1):
            for col in range(1, coly + 1):
                if bad_pxl_map[row, col] == 0:
                    nb_vector = [
                        new_matrix[row - 1, col - 1] * bad_pxl_map[row - 1, col - 1],
                        new_matrix[row - 1, col] * bad_pxl_map[row - 1, col],
                        new_matrix[row - 1, col + 1] * bad_pxl_map[row - 1, col + 1],
                        new_matrix[row, col - 1] * bad_pxl_map[row, col - 1],
                        new_matrix[row, col + 1] * bad_pxl_map[row, col + 1],
                        new_matrix[row + 1, col - 1] * bad_pxl_map[row + 1, col - 1],
                        new_matrix[row + 1, col] * bad_pxl_map[row + 1, col],
                        new_matrix[row + 1, col + 1] * bad_pxl_map[row + 1, col + 1]
                    ]
                    valid_pxl = np.sum(bad_pxl_map[row - 1:row + 1, col - 1:col + 1])
                    mean_pxl_value = np.median(nb_vector)
                    if valid_pxl == 0:
                        corr_img[row, col] = corr_img[row, col]
                    else:
                        corr_img[row, col] = mean_pxl_value

        self.bad_pxl_corr_img = corr_img[1:1025, 1:1025]
        self.bad_pxl_matrix = self.bad_pxl_matrix[1:1025, 1:1025]
        print("Bad pixel correction done...")

    def _sensitivity_correction(self):
        self.img_result = self.pxl_gain_matrix * self.bad_pxl_corr_img
        print("Sensitivity correction done...")

        self.proc_image = signal.medfilt2d(self.img_result, self.med_filt_kernel_size)

        if self.med_filt_kernel_size > 1:
            print(f"Median filter of {self.med_filt_kernel_size * self.med_filt_kernel_size * self.pxl_shift_in_mm * self.pxl_shift_in_mm:.1f} mm² kernel size applied")
        else:
            print("No median filter applied...")

        self.proc_image = np.rot90(self.proc_image, 3)
        self.proc_image = np.flipud(self.proc_image)

    def _calculate_noise_and_signal(self):
        noise_statistics = [
            np.min(self.temp_noise_per_pxl),
            np.mean(self.temp_noise_per_pxl),
            np.std(self.temp_noise_per_pxl),
            np.median(self.temp_noise_per_pxl),
            np.max(self.temp_noise_per_pxl)
        ]

        self.noise = noise_statistics[1] + (5 * noise_statistics[2])
        print(f"Pixel noise: {self.noise:.0f} counts")

        self.eff_noise = self.noise * np.sqrt(self.num_of_exp_frames)
        print(f"Effective noise: {self.eff_noise:.0f} counts")

        sig_int_array = []
        for elements in range(len(self.fr_indices_beam_on)):
            sig_sur_array = self.bg_corr_img_container[self.fr_indices_beam_on[elements]]
            sig_sur_array_filt = signal.medfilt2d(sig_sur_array, 3)
            cent_of_mass = scipy.ndimage.center_of_mass(sig_sur_array_filt)
            if np.isnan(cent_of_mass[0]) or np.isnan(cent_of_mass[1]):
                break
            sig_intensity = sig_sur_array_filt[int(cent_of_mass[0]), int(cent_of_mass[1])]
            sig_int_array.append(sig_intensity)

        self.sig_intensity = np.median(sig_int_array)
        self.avg_background = np.median(self.bg_img_pre)

        if np.isnan(self.sig_intensity):
            self.sig_intensity = np.mean(self.bg_corr_bg_summed_up)

        self.snr = self.sig_intensity / self.eff_noise

        if not np.isnan(self.avg_background):
            self.avg_background = np.mean(self.bg_corr_bg_summed_up)

        self.fsr = (self.sig_intensity + self.avg_background) / self.bit_resolution

        print(f"Signal: {self.sig_intensity + self.avg_background:.0f} counts, average background: {self.avg_background:.0f} counts, SBR: {(self.sig_intensity + self.avg_background) / self.avg_background:.0f}")
        print(f"Signal to noise ratio (SNR): {self.snr:.1f}")
        print(f"{self.fsr * 100:.1f} % of FSR")

    def _write_log(self):
        with open(self.date + "_process" + ".log", "a") as f:
            f.write(f"############ {self.date_time} #############\n")
            f.write(f"{self.file}\n")
            f.write(f"Total number of frames: {self.num_of_frames}\n")
            f.write(f"Total number of valid frames: {self.last_frame_index}\n")
            f.write(f"{self.num_of_bg_frames_pre} frames pre exposure, from index {self.fr_indices_bg_pre[0]} to {self.fr_indices_bg_pre[-1]}\n")
            f.write(f"{self.num_of_exp_frames} exposed frames from index {self.fr_indices_beam_on[0]} to {self.fr_indices_beam_on[-1]}\n")
            f.write(f"{self.num_of_bg_frames_post} frames post exposure\n")
            f.write(f"{self.bad_pxl_factor * 100:.1f} % of pixel out of tolerance to be corrected\n")
            f.write(f"Median filter of {self.med_filt_kernel_size * self.med_filt_kernel_size * self.pxl_shift_in_mm * self.pxl_shift_in_mm:.1f} mm² kernel size applied\n")
            f.write(f"Pixel noise: {self.noise:.1f} counts\n")
            f.write(f"Effective noise: {self.eff_noise:.1f} counts\n")
            f.write(f"Signal: {self.sig_intensity + self.avg_background:.0f} counts, background: {self.avg_background:.0f} counts, Signal to background ratio (SBR): {(self.sig_intensity + self.avg_background) / self.avg_background:.0f}\n")
            f.write(f"Signal to noise ratio (SNR): {self.snr:.2f}\n")
            f.write(f"{self.fsr * 100:.1f} % FSR\n\n")

    def save_tiff(self, file):
        max_value = np.max(self.proc_image)
        proc_image_16bit = self.proc_image / max_value * 65535
        conv_factor = max_value / 65535
        proc_image_16bit = np.uint16(proc_image_16bit)
        imwrite(
            file + ".tiff",
            proc_image_16bit,
            resolution=(5 * 25.4, 5 * 25.4),
            metadata={
                "ConvertTo32BitFactor": conv_factor,
                "%FSR": int(self.fsr * 100),
                "SNR": self.snr
            }
        )


def main():
    file = "2004.his"  # Replace with your actual file path
    med_filt_kernel_size = 5
    processor = XRDProcessor(file, med_filt_kernel_size)
    processor.process_frames()
    processor.save_tiff("output_file")

if __name__ == "__main__":
    main()
