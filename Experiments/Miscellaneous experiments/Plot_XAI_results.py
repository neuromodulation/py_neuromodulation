import numpy as np
import matplotlib.pyplot as plt
x_axis_data_labels = np.load(f"C:/Users/ICN_GPU/Documents/Glenn_Data/FeatureImportancedata/Captum_featurelabels.npy")
ig_nt_attr_test_norm_sum = np.load(f"C:/Users/ICN_GPU/Documents/Glenn_Data/FeatureImportancedata/Captum_2023_11_03-17_31_leave_1_sub_out_across_coh.npy")

x_axis_data = np.arange(ig_nt_attr_test_norm_sum.shape[0])



# dl_attr_test_sum = dl_attr_test.detach().cpu().numpy().sum(0)
# dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

#gs_attr_test_sum = gs_attr_test.detach().cpu().numpy().sum(0)
#gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

#fa_attr_test_sum = fa_attr_test.detach().cpu().numpy().sum(0)
#fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

width = 0.14
legends = ['Int Grads w/SmoothGrad_sq']

plt.figure(figsize=(20, 10))

ax = plt.subplot()
ax.set_title('Feature importances as ranked by Integrated Gradients with squared SmoothGrad')
ax.set_ylabel('Attributions')

FONT_SIZE = 16
plt.rc('font', size=FONT_SIZE)  # fontsize of the text sizes
plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend


sortedorder = np.argsort(ig_nt_attr_test_norm_sum[:, -1])[::-1]
ax.bar(x_axis_data, ig_nt_attr_test_norm_sum[:, -1][sortedorder], align='center', alpha=0.7, color='#A90000')
#ax.bar(x_axis_data + width, ig_attr_test_norm_sum[:, -1], width, align='center', alpha=0.8, color='#eb5e7c')
#ax.bar(x_axis_data + 2 * width, gs_attr_test_norm_sum[:, -1], width, align='center', alpha=0.8, color='#4260f5')
#ax.bar(x_axis_data + 3 * width, fa_attr_test_norm_sum[:, -1], width, align='center', alpha=1.0, color='#49ba81')
ax.autoscale_view()
plt.tight_layout()

ax.set_xticks(x_axis_data + 0.5)
ax.set_xticklabels(np.array(x_axis_data_labels)[sortedorder], rotation=45, ha='right')

plt.legend(legends, loc=3)
plt.show()