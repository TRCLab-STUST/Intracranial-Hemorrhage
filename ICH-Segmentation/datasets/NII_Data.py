import jax.numpy as jnp
import nibabel as nib


class NIIData:
    def __init__(self, paired_file, ct_range=(0, 90)):
        self._paired_file = paired_file
        self._ct_range = ct_range
        self._window_level = sum(self._ct_range) // 2
        self._window_width = abs(self._ct_range[0] - self._ct_range[1])

    def _adjust_image_window(self, _image, _slope, _intercept, bit=12, dtype=jnp.uint16):
        max_val = 2 ** bit
        _A = max_val / self._window_width * _slope
        _B = (max_val / 2) - (max_val * self._window_level / self._window_width) + _A * _intercept
        adjusted_image = _A * _image + _B
        adjusted_image = jnp.clip(adjusted_image, 0, max_val - 1)

        return adjusted_image.astype(dtype=dtype)

    def extract_paired_slices(self, bit=12, dtype=jnp.uint16):
        result = []
        for idx, nii_file in enumerate(self._paired_file):
            nii_data = nib.load(nii_file)
            nii_data_obj = nii_data.dataobj
            slices = nii_data_obj.get_unscaled()
            if idx == 0:
                slope = nii_data_obj.slope
                intercept = nii_data_obj.inter
                slices = self._adjust_image_window(slices, slope, intercept, bit=bit, dtype=dtype)

            result.append(slices)

        return result
