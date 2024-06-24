#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description - Field flatness analysis according to TG224
# Last change - 20.06.2024
# Author - J. Horn

# Import modules
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter


class ImageAnalyzer:
    def __init__(self, img, pxl_shift_in_mm, file_name, detector):
        self.img = img
        self.pxl_shift_in_mm = pxl_shift_in_mm
        self.file_name = file_name
        self.detector = detector
        self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.interp_factor = 0.002
        self.fieldsize_threshold = 0.5
        self.penumbra_threshold_low = 0.2
        self.penumbra_threshold_high = 0.8
        self.eval_area_factor = 4
        self.coord, self.length_area_to_norm_in_pxl = self._set_detector_params()

    def _set_detector_params(self):
        if self.detector == "flatpanel":
            return [-102.4, 102.4, -102.4, 102.4], 100
        elif self.detector == "reference":
            return [-125, 125, -125, 125], 100
        else:
            return [-75, 75, -75, 75], 5

    def analyze(self):
        self._calculate_image_properties()
        self._find_inflection_points()
        self._calculate_normalization_value()
        self._normalize_image()
        self._calculate_fieldsize_and_penumbra()
        self._calculate_central_axis()
        self._define_evaluation_area()
        self._calculate_statistics()
        self._plot_results()
        self._save_pdf_report()

    def _calculate_image_properties(self):
        self.size_img_vert, self.size_img_horiz = self.img.shape
        self.origin_vert = np.round(self.size_img_vert / 2)
        self.origin_horiz = np.round(self.size_img_horiz / 2)

    def _find_inflection_points(self):
        med_proj_img_vert = np.sum(self.img, axis=0)
        med_proj_img_horiz = np.sum(self.img, axis=1)

        num_diff_proj_img_vert = np.gradient(med_proj_img_vert)
        num_diff_proj_img_horiz = np.gradient(med_proj_img_horiz)

        filt_num_diff_proj_img_vert = savgol_filter(num_diff_proj_img_vert, 5, 3)
        filt_num_diff_proj_img_horiz = savgol_filter(num_diff_proj_img_horiz, 5, 3)

        x_vert = np.arange(len(filt_num_diff_proj_img_vert))
        x_horiz = np.arange(len(filt_num_diff_proj_img_horiz))

        x_inter_ver = np.arange(0, x_vert[-1], self.interp_factor)
        y_inter_ver = np.interp(x_inter_ver, x_vert, filt_num_diff_proj_img_vert)

        x_inter_hor = np.arange(0, x_horiz[-1], self.interp_factor)
        y_inter_hor = np.interp(x_inter_hor, x_horiz, filt_num_diff_proj_img_horiz)

        pos_infl_pnt_vert1 = np.concatenate(np.where(y_inter_ver == np.max(y_inter_ver)))
        pos_infl_pnt_vert2 = np.concatenate(np.where(y_inter_ver == np.min(y_inter_ver)))

        self.pxl_pos_cent_irrad_field_vert = np.round(
            (pos_infl_pnt_vert1 + pos_infl_pnt_vert2) / 2 * self.interp_factor
        )

        pos_infl_pnt_horiz1 = np.concatenate(np.where(y_inter_hor == np.max(y_inter_hor)))
        pos_infl_pnt_horiz2 = np.concatenate(np.where(y_inter_hor == np.min(y_inter_hor)))

        self.pxl_pos_cent_irrad_field_horiz = np.round(
            (pos_infl_pnt_horiz1 + pos_infl_pnt_horiz2) / 2 * self.interp_factor
        )

    def _calculate_normalization_value(self):
        self.norm_value = np.median(
            self.img[
                int(self.pxl_pos_cent_irrad_field_vert - self.length_area_to_norm_in_pxl):
                int(self.pxl_pos_cent_irrad_field_vert + self.length_area_to_norm_in_pxl),
                int(self.pxl_pos_cent_irrad_field_horiz - self.length_area_to_norm_in_pxl):
                int(self.pxl_pos_cent_irrad_field_horiz + self.length_area_to_norm_in_pxl)
            ]
        )

    def _normalize_image(self):
        self.img_rel = self.img / self.norm_value

    def _calculate_fieldsize_and_penumbra(self):
        profile_vert = self.img_rel[:, int(self.pxl_pos_cent_irrad_field_horiz)]
        profile_horiz = self.img_rel[int(self.pxl_pos_cent_irrad_field_vert), :]

        pos_vector_vert = (np.arange(len(profile_vert)) - self.pxl_pos_cent_irrad_field_vert)
        pos_vector_horiz = (np.arange(len(profile_horiz)) - self.pxl_pos_cent_irrad_field_horiz)

        self.pos_vector_vert_in_mm = pos_vector_vert * self.pxl_shift_in_mm
        self.pos_vector_horiz_in_mm = pos_vector_horiz * self.pxl_shift_in_mm

        pos_vector_vert_inter = np.arange(pos_vector_vert[0], pos_vector_vert[-1], self.interp_factor)
        profile_vert_inter = np.interp(pos_vector_vert_inter, pos_vector_vert, profile_vert)

        pos_vector_horiz_inter = np.arange(pos_vector_horiz[0], pos_vector_horiz[-1], self.interp_factor)
        profile_horiz_inter = np.interp(pos_vector_horiz_inter, pos_vector_horiz, profile_horiz)

        thres_vert = profile_vert_inter >= self.fieldsize_threshold
        vert_positions = np.concatenate(np.where(thres_vert == 1))
        fieldsize_vert = (vert_positions[-1] - vert_positions[0]) * self.interp_factor
        self.fs_vert_in_mm = fieldsize_vert * self.pxl_shift_in_mm

        thres_horiz = profile_horiz_inter >= self.fieldsize_threshold
        horiz_positions = np.concatenate(np.where(thres_horiz == 1))
        fieldsize_horiz = (horiz_positions[-1] - horiz_positions[0]) * self.interp_factor
        self.fs_horiz_in_mm = fieldsize_horiz * self.pxl_shift_in_mm

        thres20_vert = profile_vert_inter >= self.penumbra_threshold_low
        thres80_vert = profile_vert_inter >= self.penumbra_threshold_high
        thres20_vert = np.concatenate(np.where(thres20_vert == 1))
        thres80_vert = np.concatenate(np.where(thres80_vert == 1))

        self.penumbra_u = (thres80_vert[0] - thres20_vert[0]) * self.interp_factor
        self.penumbra_d = (thres20_vert[-1] - thres80_vert[-1]) * self.interp_factor

        thres20_horiz = profile_horiz_inter >= self.penumbra_threshold_low
        thres80_horiz = profile_horiz_inter >= self.penumbra_threshold_high
        thres20_horiz = np.concatenate(np.where(thres20_horiz == 1))
        thres80_horiz = np.concatenate(np.where(thres80_horiz == 1))

        self.penumbra_l = (thres80_horiz[0] - thres20_horiz[0]) * self.interp_factor
        self.penumbra_r = (thres20_horiz[-1] - thres80_horiz[-1]) * self.interp_factor

    def _calculate_central_axis(self):
        self.cax_horiz = (self.pxl_pos_cent_irrad_field_horiz - self.origin_horiz) * self.pxl_shift_in_mm
        self.cax_vert = (self.pxl_pos_cent_irrad_field_vert - self.origin_vert) * self.pxl_shift_in_mm

    def _define_evaluation_area(self):
        min_penumbra = np.min([self.penumbra_l, self.penumbra_r, self.penumbra_u, self.penumbra_d])
        min_fieldsize = np.min([self.fs_vert_in_mm, self.fs_horiz_in_mm])

        eval_range = int(min_fieldsize - (self.eval_area_factor * min_penumbra))

        horiz_start = int(self.pxl_pos_cent_irrad_field_horiz - (eval_range / 2))
        horiz_end = int(self.pxl_pos_cent_irrad_field_horiz + (eval_range / 2))
        self.horiz = np.arange(horiz_start, horiz_end)

        vert_start = int(self.pxl_pos_cent_irrad_field_vert - (eval_range / 2))
        vert_end = int(self.pxl_pos_cent_irrad_field_vert + (eval_range / 2))
        self.vert = np.arange(vert_start, vert_end)

    def _calculate_statistics(self):
        img_rel_roi = self.img_rel[self.vert[0]:self.vert[-1], self.horiz[0]:self.horiz[-1]]

        self.d_max = np.max(img_rel_roi)
        self.d_min = np.min(img_rel_roi)
        self.d_range = self.d_max - self.d_min
        self.d_mean = np.mean(img_rel_roi)
        self.d_sd = np.std(img_rel_roi)

        q1 = 0.001
        q2 = 1 - q1
        sorted_array = np.sort(img_rel_roi.flatten())

        self.qa = sorted_array[int(np.round(q1 * len(sorted_array)))]
        self.qb = sorted_array[int(np.round(q2 * len(sorted_array)))]
        self.d_qb_minus_qa = self.qb - self.qa
        self.d_iqr_percent = (self.d_qb_minus_qa / 2) * 100

        self.flatness_tg224 = (self.d_max - self.d_min) / (self.d_max + self.d_min)
        self.flatness_percent = self.flatness_tg224 * 100

        print(f"\n2D-Flatness: {self.flatness_percent:.1f} %...")

    def _plot_results(self):
        rows, cols = np.shape(self.img)
        img_2d_plot = np.ones([rows, cols]) * self.img_rel

        img_2d_plot[self.vert[0]:self.vert[-1], self.horiz[0]:self.horiz[1]] = 0
        img_2d_plot[self.vert[0]:self.vert[-1], self.horiz[-2]:self.horiz[-1]] = 0
        img_2d_plot[self.vert[0]:self.vert[1], self.horiz[0]:self.horiz[-1]] = 0
        img_2d_plot[self.vert[-2]:self.vert[-1], self.horiz[0]:self.horiz[-1]] = 0
        img_2d_plot[int(self.pxl_pos_cent_irrad_field_vert), :] = 0
        img_2d_plot[:, int(self.pxl_pos_cent_irrad_field_horiz)] = 0

        fig, axes = plt.subplots(figsize=(10, 10), dpi=150)

        plt.rcParams['axes.titlesize'] = 12
        axes.tick_params(axis='x', labelsize=12)
        axes.tick_params(axis='y', labelsize=12)

        img_handle = axes.imshow(
            img_2d_plot, cmap=plt.cm.bone, vmin=0.8, vmax=1.2,
            origin="lower", extent=self.coord
        )
        axes.set(title=f"2D-Flatness within ROI = ± {self.flatness_percent:.1f}%")
        axes.set_xlabel("horizontal Position [mm]", fontsize=12)
        axes.set_ylabel("vertical Position [mm]", fontsize=12)
        cbar = fig.colorbar(img_handle, shrink=0.5)
        cbar.set_label("normalized Signal", fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        axes.set_xlim(-75, 75)
        axes.set_ylim(-75, 75)

        fig1, axes1 = plt.subplots(2, 1, figsize=(10, 10), dpi=150)
        plt.rcParams['axes.titlesize'] = 12

        n_bins = int((max(sorted_array) - min(sorted_array)) / 0.001)
        counts, bins = np.histogram(sorted_array, bins=n_bins)
        axes1[0].hist(sorted_array, bins, weights=np.ones_like(sorted_array) / len(sorted_array) * 100,
                      color="white", ec="black")
        axes1[0].vlines(x=self.d_mean, ymin=0, ymax=20, color="black", linestyle="-")
        axes1[0].vlines(x=[self.d_mean + self.d_sd, self.d_mean - self.d_sd], ymin=0, ymax=20,
                        color="green", linestyle="--")
        axes1[0].vlines(x=[self.d_max, self.d_min], ymin=0, ymax=20, color="red", linestyle=":")
        axes1[0].vlines(x=[0.95, 1.05], ymin=0, ymax=20, color="red")
        axes1[0].set(title="Histogram of normalized Signal within ROI")
        axes1[0].set_xlabel("normalized Signal", fontsize=12)
        axes1[0].set_ylabel("Frequency [%]", fontsize=12)
        axes1[0].set(xlim=(0.94, 1.06))
        axes1[0].set_xticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06])
        axes1[0].set(ylim=(0, 10))
        axes1[0].legend(
            [f"μ = {self.d_mean:.2f}", f"σ = ± {self.d_sd:.2f}", f"Dmax - Dmin = {self.d_range:.2f}",
             "Tolerance-Levels"], loc="upper right", framealpha=1)

        axes1[1].plot(self.pos_vector_vert_in_mm, profile_vert, "g.", self.pos_vector_horiz_in_mm, profile_horiz, "b.")
        axes1[1].set(title="Central Line Profile")
        axes1[1].set_xlabel("Position [mm]", fontsize=12)
        axes1[1].set_ylabel("normalized Signal", fontsize=12)
        axes1[1].set(ylim=(0.5, 1.2))
        axes1[1].set(xlim=(-75, 75))
        axes1[1].legend([f"vertical Fieldsize = {self.fs_vert_in_mm:.1f} mm",
                         f"horizontal Fieldsize = {self.fs_horiz_in_mm:.1f} mm"],
                        loc="center", framealpha=1)
        axes1[1].axhline(0.95, color="red")
        axes1[1].axhline(1.05, color="red")
        axes1[1].grid(True)

        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10), dpi=150)
        pos_horiz = (self.horiz - self.pxl_pos_cent_irrad_field_horiz) * self.pxl_shift_in_mm
        pos_vert = (self.vert - self.pxl_pos_cent_irrad_field_vert) * self.pxl_shift_in_mm

        plt.rcParams['axes.titlesize'] = 12
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)

        axes2[0].plot(pos_horiz, self.img_rel[self.vert, self.horiz[0]:self.horiz[-1]])
        axes2[0].set(title="Line profiles")
        axes2[0].set_xlabel("vertical Position [mm]", fontsize=12)
        axes2[0].set_ylabel("normalized Signal", fontsize=12)
        axes2[0].set(ylim=(0.8, 1.2))
        axes2[0].set(xlim=(-75, 75))
        axes2[0].axhline(0.95, color="red")
        axes2[0].axhline(1.05, color="red")
        axes2[0].grid(True)

        img_rel_trans = np.transpose(self.img_rel)
        axes2[1].plot(pos_vert, img_rel_trans[self.horiz, self.vert[0]:self.vert[-1]])
        axes2[1].set_xlabel("horizontal Position [mm]", fontsize=12)
        axes2[1].set_ylabel("normalized Signal", fontsize=12)
        axes2[1].set(ylim=(0.8, 1.2))
        axes2[1].set(xlim=(-75, 75))
        axes2[1].axhline(0.95, color="red")
        axes2[1].axhline(1.05, color="red")
        axes2[1].grid(True)

    def _save_pdf_report(self):
        plt.rcParams["figure.figsize"] = [11.69, 8.268]
        plt.rcParams["figure.autolayout"] = False
        horiz_line = "__________________________________________________"
        p = PdfPages(self.file_name)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        tot_nb_pages = len(figs)
        count = 0

        for fig in figs:
            count += 1
            fig.text(0.05, 0.99, horiz_line)
            fig.text(0.05, 0.97, self.file_name)
            fig.text(0.05, 0.02, self.date)
            fig.text(0.8, 0.02, f"Page {count} of {tot_nb_pages}")
            fig.savefig(p, format="pdf", dpi=150)
            plt.close()
        p.close()


def main():
    # Example usage
    img = np.random.rand(2048, 2048)  # Replace with your actual image
    pxl_shift_in_mm = 0.1
    file_name = "output.pdf"
    detector = "flatpanel"

    analyzer = ImageAnalyzer(img, pxl_shift_in_mm, file_name, detector)
    analyzer.analyze()


if __name__ == "__main__":
    main()
