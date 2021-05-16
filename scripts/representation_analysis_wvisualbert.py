import os
import argparse
import numpy as np
import pickle
import json
import glob
import matplotlib
import matplotlib.pyplot as plt

from pytorch_transformers.tokenization_bert import BertTokenizer
from sklearn.manifold import TSNE
#import umap

stopwords = [line.strip('\n') for line in open('stopwords.txt')]
GLOBAL_EMBEDDING_V_LABEL = '[IMG]'
GLOBAL_EMBEDDING_T_LABEL = '[CLS]'
MASK_TOKEN_T = 103
RND_TOKEN_T = 99999
TEXT_INPUT = 0
OBJ_TAG_INPUT = 1
VISUAL_INPUT = 2

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--pkl_dir", default="/Users/julius/Projects/logs/conceptual_captions/mixuniter_obj1_sharedKV_base", type=str)
    parser.add_argument("--vg_categories", default="../config/visual_genome_categories.json", type=str)

    # Text
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, ...")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--overlapped_tokens_first", action='store_true')
    parser.add_argument("--step", type=int)
    parser.add_argument("--topk_t", default=100, type=int)
    parser.add_argument("--topk_v", default=25, type=int)
    parser.add_argument("--num_t", default=-1, type=int)
    parser.add_argument("--num_v", default=-1, type=int)
    parser.add_argument("--num_tag", default=-1, type=int)
    parser.add_argument("--num_extra_text_tokens", default=0, type=int)
    parser.add_argument("--hide_masked_tokens", default=False, action="store_true")
    parser.add_argument("--layer_of_interest", default=3, type=int)

    parser.add_argument("--global_emb", default="first_token", type=str)     # or 'pooled'
    parser.add_argument("--global_emb_v_label", default="[IMG]", type=str)
    parser.add_argument("--global_emb_t_label", default="[SENT]", type=str)

    return parser.parse_args()

def top_visual_classes(args, vg_categories, topk=50):
    obj_stats_filepath = os.path.join(args.pkl_dir, 'sorted_obj_counts.pkl')
    obj_stats = pickle.load(open(obj_stats_filepath, 'rb'))

    topk_classes = list(obj_stats.keys())[:topk]
    for key in topk_classes:
        print('{:>8} ({:>5}) {:>8}'.format(vg_categories[key], key, obj_stats[key]), flush=True)

    return topk_classes

# ----------------------------------------------------------------------
def plot_global_embedding(tsne_global_embeddings, global_embeddings_labels, global_embeddings_discriminator_scores,
                          at_step, model_name, title):
    plt.figure()
    plt.title(title)

    tsne_global_embeddings_t = tsne_global_embeddings[global_embeddings_labels == '[CLS]']
    if tsne_global_embeddings_t.shape[0] == 0:
        tsne_global_embeddings_t = tsne_global_embeddings[global_embeddings_labels == '[SENT]']
        global_embeddings_discriminator_scores_t = global_embeddings_discriminator_scores[
            global_embeddings_labels == '[SENT]']
    else:
        global_embeddings_discriminator_scores_t = global_embeddings_discriminator_scores[
            global_embeddings_labels == '[CLS]']

    tsne_global_embeddings_v = tsne_global_embeddings[global_embeddings_labels == '[IMG]']
    global_embeddings_discriminator_scores_v = global_embeddings_discriminator_scores[global_embeddings_labels == '[IMG]']

    markersize=8
    plt.plot(tsne_global_embeddings_v[0, 0], tsne_global_embeddings_v[0, 1], 'ko', label='visual',
             fillstyle='full', markeredgewidth=0.0, markersize=markersize, alpha=0.5)
    plt.plot(tsne_global_embeddings_t[0, 0], tsne_global_embeddings_t[0, 1], 'k<', label='textual',
             fillstyle='full', markeredgewidth=0.0, markersize=markersize, alpha=0.5)

    plt.plot(tsne_global_embeddings_v[:, 0][global_embeddings_discriminator_scores_v < 0.5],
             tsne_global_embeddings_v[:, 1][global_embeddings_discriminator_scores_v < 0.5],
             MARKER_SHAPE_V, color='r', fillstyle='full', markeredgewidth=0.0, markersize=markersize, alpha=0.5)
    plt.plot(tsne_global_embeddings_v[:, 0][global_embeddings_discriminator_scores_v >= 0.5],
             tsne_global_embeddings_v[:, 1][global_embeddings_discriminator_scores_v >= 0.5],
             MARKER_SHAPE_V, color='b', fillstyle='full', markeredgewidth=0.0, markersize=markersize, alpha=0.5)

    plt.plot(tsne_global_embeddings_t[:, 0][global_embeddings_discriminator_scores_t < 0.5],
             tsne_global_embeddings_t[:, 1][global_embeddings_discriminator_scores_t < 0.5],
             '<', color='r', fillstyle='full', markeredgewidth=0.0, markersize=markersize, alpha=0.5)
    plt.plot(tsne_global_embeddings_t[:, 0][global_embeddings_discriminator_scores_t >= 0.5],
             tsne_global_embeddings_t[:, 1][global_embeddings_discriminator_scores_t >= 0.5],
             '<', color='b', fillstyle='full', markeredgewidth=0.0, markersize=markersize, alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig('{}_global_embeddings_{}.png'.format(model_name, at_step), dpi=150)


# Scale and visualize the embedding vectors
def plot_embedding(X, labels, domain_labels, mask_labels, at_step, topk_t=25, topk_v=25, title=None, layer_of_interest=12, model_name='knnkl',
                   num_v=-1, num_t=-1, num_tag=-1, overlapped_tokens_first=False, num_extra_text_tokens=0, hide_masked_tokens=False):

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    X_tag = X[domain_labels == OBJ_TAG_INPUT]
    X_v = X[domain_labels == VISUAL_INPUT]
    X_t = X[domain_labels == TEXT_INPUT]

    labels_tag = labels[domain_labels == OBJ_TAG_INPUT]
    labels_v = labels[domain_labels == VISUAL_INPUT]
    labels_t = labels[domain_labels == TEXT_INPUT]

    mask_labels_tag = mask_labels[domain_labels == OBJ_TAG_INPUT]
    mask_labels_v = mask_labels[domain_labels == VISUAL_INPUT]
    mask_labels_t = mask_labels[domain_labels == TEXT_INPUT]

    def get_topk_labels(labels, topk, always_include_CLS_IMG=False):
        unique, counts = np.unique(labels, return_counts=True)
        counts = dict(zip(unique, counts))
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])}
        topk_list = list(counts.keys())[0:topk]

        if always_include_CLS_IMG:
            for special_token in ['[IMG]', '[CLS]', '[SENT]']:
                if special_token in counts and special_token not in topk_list:
                    topk_list.append(special_token)

        return topk_list

    topk_labels_v = get_topk_labels(labels_v, topk=topk_v)
    topk_labels_t = get_topk_labels(labels_t, topk=topk_t)

    print('top {} visual labels: {}'.format(topk_v, topk_labels_v))
    print('top {} textual labels: {}'.format(topk_t, topk_labels_t))

    if overlapped_tokens_first:
        topk_labels_v = list(set(topk_labels_v) & set(topk_labels_t))
        topk_labels_t_expand = []
        extra_count = 0
        for l in topk_labels_t:
            # skip these
            if l in ['[IMG]', '[CLS]', '[SENT]']:
                continue
            if l in topk_labels_v:
                continue
            topk_labels_t_expand.append(l)
            extra_count += 1
            if extra_count == num_extra_text_tokens:
                break

        topk_labels_t_expand += topk_labels_v
        topk_labels_t = list(topk_labels_t_expand)

    print('these visual tokens will be visualized: {}'.format(topk_labels_v))
    print('these textual tokens will be visualized: {}'.format(topk_labels_t))

    selected_labels_v = np.array([True if l in topk_labels_v else False for l in labels_v])
    selected_labels_t = np.array([True if l in topk_labels_t else False for l in labels_t])
    selected_labels_tag = np.array([True if l in topk_labels_t else False for l in labels_tag])

    X_tag = X_tag[selected_labels_tag]
    X_v = X_v[selected_labels_v]
    X_t = X_t[selected_labels_t]

    labels_tag = labels_tag[selected_labels_tag]
    labels_v = labels_v[selected_labels_v]
    labels_t = labels_t[selected_labels_t]

    mask_labels_tag = mask_labels_tag[selected_labels_tag]
    mask_labels_v = mask_labels_v[selected_labels_v]
    mask_labels_t = mask_labels_t[selected_labels_t]

    if num_v > 0:
        num_v = min(num_v, selected_labels_v.shape[0])
        selected_ind = np.random.choice(np.arange(labels_v.shape[0]), num_v)
        X_v = X_v[selected_ind]
        labels_v = labels_v[selected_ind]
        mask_labels_v = mask_labels_v[selected_ind]

    if num_t > 0:
        num_t = min(num_t, selected_labels_t.shape[0])
        selected_ind = np.random.choice(np.arange(labels_t.shape[0]), num_t)
        X_t = X_t[selected_ind]
        labels_t = labels_t[selected_ind]
        mask_labels_t = mask_labels_t[selected_ind]

    if num_tag > 0:
        num_tag = min(num_tag, selected_labels_tag.shape[0])
        selected_ind = np.random.choice(np.arange(labels_tag.shape[0]), num_tag)
        X_tag = X_tag[selected_ind]
        labels_tag = labels_tag[selected_ind]
        mask_labels_tag = mask_labels_tag[selected_ind]

    def convert(set):
        return list(set)

    # generate color map
    set_labels_v = sorted(set(labels_v))
    set_labels_t = sorted(set(labels_t))
    set_labels_tag = sorted(set(labels_tag))
    set_labels = set(convert(set_labels_v) + convert(set_labels_t) + convert(set_labels_tag))
    Y_colors = len(set_labels)
    Y_label_color = {y: float(i) / Y_colors for i, y in enumerate(set_labels)}

    plt.figure(figsize=(18, 9))
    # first plot: tokens / visual classes and their 2D embeddings
    ax = plt.subplot(121)
    #fontdict = {'weight': 'bold', 'size': 9}
    fontdict = {'size': 9}

    '''min_x = min(np.min(X_v[:, 0]), np.min(X_t[:, 0])) - 0.025
    max_x = max(np.max(X_v[:, 0]), np.max(X_t[:, 0])) + 0.025

    min_y = min(np.min(X_v[:, 1]), np.min(X_t[:, 1])) - 0.025
    max_y = max(np.max(X_v[:, 1]), np.max(X_t[:, 1])) + 0.025'''

    min_x = -0.025
    max_x = 1.025
    min_y = min_x
    max_y = max_x

    CM = plt.cm.nipy_spectral
    for i in range(X_v.shape[0]):
        color = CM(Y_label_color[labels_v[i]])
        plt.text(X_v[i, 0], X_v[i, 1], str(labels_v[i]), color=color, fontdict=fontdict)

    for i in range(X_t.shape[0]):
        color = CM(Y_label_color[labels_t[i]])
        plt.text(X_t[i, 0], X_t[i, 1], str(labels_t[i]), color=color, fontdict=fontdict)

    for i in range(X_tag.shape[0]):
        color = CM(Y_label_color[labels_tag[i]])
        plt.text(X_tag[i, 0], X_tag[i, 1], str(labels_tag[i]), color=color, fontdict=fontdict)

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    plt.subplot(122)
    markersize = 8
    #plt.plot(X_v[:, 0], X_v[:, 1], 'ko', label='visual', alpha=0.5, markeredgewidth=0.0, markersize=markersize)
    #plt.plot(X_t[:, 0], X_t[:, 1], 'k<', label='textual', alpha=0.5, markeredgewidth=0.0, markersize=markersize)

    ALPHA_NOTMASKED_TOKENS=0.5
    ALPHA_MASKED_TOKENS=0.15
    MARKER_SHAPE_V = 'o'
    MARKER_SHAPE_T = '<'
    MARKER_SHAPE_TAG = 's'
    MASKED_COLOR_K = matplotlib.colors.colorConverter.to_rgba('k', alpha=ALPHA_MASKED_TOKENS)
    MASKED_COLOR_B = matplotlib.colors.colorConverter.to_rgba('b', alpha=ALPHA_MASKED_TOKENS)
    for label in Y_label_color.keys():
        if label in set_labels_v:
            color = CM(Y_label_color[label])
            x = X_v[labels_v == label]
            mask_labels_target = mask_labels_v[labels_v == label]
            x_notmasked = x[mask_labels_target == -1]
            x_masked = x[mask_labels_target != -1]
            plt.plot(x_notmasked[:, 0], x_notmasked[:, 1], MARKER_SHAPE_V, color=color, alpha=ALPHA_NOTMASKED_TOKENS,
                     fillstyle='full', markeredgecolor=color, markeredgewidth=0.0, markersize=markersize)
            plt.plot(x_masked[:, 0], x_masked[:, 1], MARKER_SHAPE_V, color='k',
                     fillstyle='none', markeredgecolor='k', markersize=markersize, alpha=ALPHA_MASKED_TOKENS)

            if x_masked.shape[0] > 0:
                template_v_masked = x_masked[0]
            if x_notmasked.shape[0] > 0:
                template_v_notmasked = x_notmasked[0]

        if label in set_labels_t:
            color = CM(Y_label_color[label])
            x = X_t[labels_t == label]
            mask_labels_target = mask_labels_t[labels_t == label]
            x_notmasked = x[mask_labels_target == -1]
            x_masked = x[mask_labels_target == MASK_TOKEN_T]
            x_random = x[mask_labels_target == RND_TOKEN_T]

            # some tokens are neither masked nor randomly replaced!
            x_original_tokens = x[(mask_labels_target != -1) & (mask_labels_target != MASK_TOKEN_T) &
                                  (mask_labels_target != RND_TOKEN_T)]

            plt.plot(x_notmasked[:, 0], x_notmasked[:, 1], MARKER_SHAPE_T, color=color, alpha=ALPHA_NOTMASKED_TOKENS,
                     fillstyle='full', markeredgecolor=color, markeredgewidth=0.0, markersize=markersize)
            plt.plot(x_original_tokens[:, 0], x_original_tokens[:, 1], MARKER_SHAPE_T, color=color, alpha=ALPHA_NOTMASKED_TOKENS,
                     fillstyle='full', markeredgecolor=color, markeredgewidth=0.0, markersize=markersize)
            plt.plot(x_masked[:, 0], x_masked[:, 1], MARKER_SHAPE_T,
                     fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, alpha=ALPHA_MASKED_TOKENS)
            plt.plot(x_random[:, 0], x_random[:, 1], MARKER_SHAPE_T,
                     fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, alpha=ALPHA_MASKED_TOKENS)

            if x_masked.shape[0] > 0:
                template_t_masked = x_masked[0]
            if x_notmasked.shape[0] > 0:
                template_t_notmasked = x_notmasked[0]
            if x_random.shape[0] > 0:
                template_t_random = x_random[0]

        if label in set_labels_tag:
            color = CM(Y_label_color[label])
            x = X_tag[labels_tag == label]
            mask_labels_target = mask_labels_tag[labels_tag == label]
            x_notmasked = x[mask_labels_target == -1]
            x_masked = x[mask_labels_target == MASK_TOKEN_T]
            x_random = x[mask_labels_target == RND_TOKEN_T]

            # some tokens are neither masked nor randomly replaced!
            x_original_tokens = x[(mask_labels_target != -1) & (mask_labels_target != MASK_TOKEN_T) &
                                  (mask_labels_target != RND_TOKEN_T)]

            plt.plot(x_notmasked[:, 0], x_notmasked[:, 1], MARKER_SHAPE_TAG, color=color, alpha=ALPHA_NOTMASKED_TOKENS,
                     fillstyle='full', markeredgecolor=color, markeredgewidth=0.0, markersize=markersize)
            plt.plot(x_original_tokens[:, 0], x_original_tokens[:, 1], MARKER_SHAPE_TAG, color=color, alpha=ALPHA_NOTMASKED_TOKENS,
                     fillstyle='full', markeredgecolor=color, markeredgewidth=0.0, markersize=markersize)
            plt.plot(x_masked[:, 0], x_masked[:, 1], MARKER_SHAPE_TAG,
                     fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, alpha=ALPHA_MASKED_TOKENS)
            plt.plot(x_random[:, 0], x_random[:, 1], MARKER_SHAPE_TAG,
                     fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, alpha=ALPHA_MASKED_TOKENS)

            if x_masked.shape[0] > 0:
                template_tag_masked = x_masked[0]
            if x_notmasked.shape[0] > 0:
                template_tag_notmasked = x_notmasked[0]
            if x_random.shape[0] > 0:
                template_tag_random = x_random[0]

    if 'template_v_notmasked' in locals():
        plt.plot(template_v_notmasked[0], template_v_notmasked[1], MARKER_SHAPE_V, color='k', alpha=ALPHA_NOTMASKED_TOKENS,
                 fillstyle='full', markeredgecolor='k', markeredgewidth=0.0, markersize=markersize, label='visual')
    if 'template_t_notmasked' in locals():
        plt.plot(template_t_notmasked[0], template_t_notmasked[1], MARKER_SHAPE_T, color='k', alpha=ALPHA_NOTMASKED_TOKENS,
                 fillstyle='full', markeredgecolor='k', markeredgewidth=0.0, markersize=markersize, label='textual')
    if 'template_tag_notmasked' in locals():
        plt.plot(template_tag_notmasked[0], template_tag_notmasked[1], MARKER_SHAPE_TAG, color='k', alpha=ALPHA_NOTMASKED_TOKENS,
                 fillstyle='full', markeredgecolor='k', markeredgewidth=0.0, markersize=markersize, label='obj_tags')

    if 'template_v_masked' in locals():
        plt.plot(template_v_masked[0], template_v_masked[1], MARKER_SHAPE_V, color='k',
                 fillstyle='none', markeredgecolor=MASKED_COLOR_K, markersize=markersize, label='visual (masked)', alpha=ALPHA_MASKED_TOKENS)
    if 'template_t_masked' in locals():
        plt.plot(template_t_masked[0], template_t_masked[1], MARKER_SHAPE_T, color='k',
                 fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, label='textual (masked/random)', alpha=ALPHA_MASKED_TOKENS)
    #if 'template_t_random' in locals():
    #    plt.plot(template_t_random[0], template_t_random[1], MARKER_SHAPE_T,
    #             fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, label='textual (masked/random)', alpha=ALPHA_MASKED_TOKENS)

    if 'template_tag_masked' in locals():
        plt.plot(template_tag_masked[0], template_v_masked[1], MARKER_SHAPE_V, color='k',
                 fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, label='obj_tags (masked/random)', alpha=ALPHA_MASKED_TOKENS)
    #if 'template_tag_random' in locals():
    #    plt.plot(template_tag_random[0], template_tag_random[1], MARKER_SHAPE_TAG,
    #             fillstyle='none', markeredgecolor=MASKED_COLOR_B, markersize=markersize, label='obj_tags (masked/random)', alpha=ALPHA_MASKED_TOKENS)

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if title is not None:
        #plt.title(title)
        plt.suptitle(title)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    if overlapped_tokens_first:
        plt.savefig('{}_overlapped_VLtokens_{}_{}.png'.format(model_name, layer_of_interest, at_step), dpi=150)
    else:
        plt.savefig('{}_analyze_word_from_layer_{}_{}.png'.format(model_name, layer_of_interest, at_step), dpi=150)
    #plt.show()

def analyze_word_from_layers(args, tokenizer, files, topk_classes, layer_of_interest, vg_categories, model_name,
                             num_t, num_v, num_tag):
    '''

    :param files:
    :param topk_classes:
    :param layer_of_interest:   0, 1, 2, 3; 3: top layer
    :return:
    '''

    embedding_filepath = '{}_analyze_word_from_layer_{}_embeddings_{}.pkl'.format(model_name, layer_of_interest, args.step)
    tsne_embedding_filepath = '{}_analyze_word_from_layer_{}_tsne_embeddings_{}.pkl'.format(model_name, layer_of_interest, args.step)

    if os.path.exists(embedding_filepath):
        print('found cached embeddings at [{}]'.format(embedding_filepath))
        data = pickle.load(open(embedding_filepath, 'rb'))
        embeddings = data['embeddings']
        labels = data['labels']
        domain_labels = data['domain_labels']
        mask_labels = data['mask_labels']
    else:
        embeddings_v = []
        labels_v = []
        mask_labels_v = []  # 1: masked, 0: not masked

        embeddings_t = []
        labels_t = []
        mask_labels_t = []  # 1: masked, 0: not masked
        domain_t = []

        print('collecting visual embeddings...')
        cat_embeddings = {}
        for file in files:
            print('file: {}'.format(file))
            data = pickle.load(open(file, 'rb'))

            obj_labels = data['obj_labels']     # (256, 36)
            local_embeddings_tv_v = data['sequence_output_tv_vx'][layer_of_interest][:, 1::]  # (batch, n_tokens, dim), visual tokens

            for cat in topk_classes:
                embeddings_tv_v_target_class = local_embeddings_tv_v[obj_labels == cat]
                mask_tv_v_target_class = data['image_label'][obj_labels == cat]
                assert mask_tv_v_target_class.shape[0] == embeddings_tv_v_target_class.shape[0]

                if embeddings_tv_v_target_class.shape[0] > 0:
                    if cat in cat_embeddings:
                        cat_embeddings[cat]['embeddings'].append(embeddings_tv_v_target_class)
                        cat_embeddings[cat]['mask_labels'].append(mask_tv_v_target_class)
                    else:
                        cat_embeddings[cat] = {'embeddings': [embeddings_tv_v_target_class],
                                               'mask_labels': [mask_tv_v_target_class]}

        for cat in cat_embeddings:
            for key in cat_embeddings[cat]:
                cat_embeddings[cat][key] = np.concatenate(cat_embeddings[cat][key])     # (n, dim)

            embeddings_v.append(cat_embeddings[cat]['embeddings'])
            labels_v.append([vg_categories[cat]] * cat_embeddings[cat]['embeddings'].shape[0])
            mask_labels_v.append(cat_embeddings[cat]['mask_labels'])

        embeddings_v = np.concatenate(embeddings_v)
        labels_v = np.concatenate(labels_v)
        mask_labels_v = np.concatenate(mask_labels_v)
        domain_v = np.ones_like(labels_v, dtype=np.int8) * VISUAL_INPUT

        print('collecting language embeddings...')
        for file in files:
            print('file: {}'.format(file))
            data = pickle.load(open(file, 'rb'))

            masked_obj_labels = data['lm_label_ids']    # (batch, n_tokens including [CLS] and [SEP])
            tokens_t = data['input_ids']                # (batch, n_tokens including [CLS] and [SEP])

            local_embeddings_t = data['sequence_output_tx'][layer_of_interest]  # (batch, n_tokens, dim), language tokens
            local_embeddings_tv_t = data['sequence_output_tv_tx'][layer_of_interest]  # (batch, n_tokens, dim), object tags
            obj_tokens = data['obj_tokens']  # (256, 64)
            obj_tokens_label = data['obj_tokens_label']  # (256, 64)

            def process_text_type_tokens(tokens_t, masked_obj_labels, embeddings, input_type):
                all_tokens = [tokenizer.convert_ids_to_tokens(tokens) for tokens in tokens_t]
                for i, tokens in enumerate(all_tokens):
                    for j, token in enumerate(tokens):
                        if token == '[MASK]':
                            # keep the original token before masked
                            token = tokenizer.convert_ids_to_tokens([masked_obj_labels[i][j]])[0]
                            masked_obj_labels[i][j] = MASK_TOKEN_T
                        else:
                            if masked_obj_labels[i][j] != -1:
                                # keep the original token before being replaced by a random token
                                ori_token = tokenizer.convert_ids_to_tokens([masked_obj_labels[i][j]])[0]
                                if ori_token != token:
                                    masked_obj_labels[i][j] = RND_TOKEN_T
                                token = ori_token

                        if token in stopwords:
                            continue
                        if '#' in token:
                            continue

                        labels_t.append(token)
                        embeddings_t.append(embeddings[i][j])
                        mask_labels_t.append(masked_obj_labels[i][j])
                        domain_t.append(input_type)

            # texts
            process_text_type_tokens(tokens_t, masked_obj_labels, local_embeddings_t, TEXT_INPUT)
            # object tags
            process_text_type_tokens(obj_tokens, obj_tokens_label, local_embeddings_tv_t, OBJ_TAG_INPUT)

        embeddings_t = np.array(embeddings_t)
        labels_t = np.array(labels_t)
        mask_labels_t = np.array(mask_labels_t)
        domain_t = np.array(domain_t)

        embeddings = np.concatenate([embeddings_t, embeddings_v])
        labels = np.concatenate([labels_t, labels_v])
        domain_labels = np.concatenate([domain_t, domain_v])
        mask_labels = np.concatenate([mask_labels_t, mask_labels_v])

        with open(embedding_filepath, 'wb') as fp:
            pickle.dump({'embeddings': embeddings, 'labels': labels, 'domain_labels': domain_labels,
                         'mask_labels': mask_labels}, fp)

    print('embeddings {} labels {} domain_labels {} mask_labels {} '.format(embeddings.shape, labels.shape,
                                                                            domain_labels.shape, mask_labels.shape))

    if os.path.exists(tsne_embedding_filepath):
        print('found cached tsne embeddings at [{}]'.format(tsne_embedding_filepath))
        data = pickle.load(open(tsne_embedding_filepath, 'rb'))
        tsne_embeddings = data['tsne_embeddings']
    else:
        print('not found cached embeddings, re-calculating...')
        tsne_embeddings = TSNE(n_components=2, init='pca', n_jobs=24, verbose=1).fit_transform(embeddings)
        #tsne_embeddings = umap.UMAP(n_components=2, n_jobs=8).fit_transform(embeddings)
        with open(tsne_embedding_filepath, 'wb') as fp:
            pickle.dump({'tsne_embeddings': tsne_embeddings}, fp)

    print('reduced embeddings {}'.format(tsne_embeddings.shape))
    assert tsne_embeddings.shape[0] == embeddings.shape[0]

    layers = {0:0, 1:4, 2:8, 3:11}
    plot_embedding(tsne_embeddings, labels, domain_labels, mask_labels,
                   title='Model: {}, vision & language t-SNE embeddings, layer {}'.format(model_name, layers[layer_of_interest] + 1),
                   at_step=args.step, topk_t=args.topk_t, topk_v=args.topk_v, layer_of_interest=layers[layer_of_interest] + 1,
                   model_name=model_name, num_t=num_t, num_v=num_v, num_tag=num_tag,
                   overlapped_tokens_first=args.overlapped_tokens_first, num_extra_text_tokens=args.num_extra_text_tokens,
                   hide_masked_tokens=args.hide_masked_tokens)

def analyze_sentence_from_layer(files):
    pass

def main():
    args = parse_args()
    model_name = os.path.basename(args.pkl_dir)
    vg_categories = json.load(open(args.vg_categories, 'rb'))['categories']
    vg_categories = {cat['id']: cat['name'] for cat in vg_categories}

    topk_classes = top_visual_classes(args, vg_categories, topk=args.topk_v)
    files = [file for file in glob.glob(os.path.join(args.pkl_dir, '0*_step_{}.pkl'.format(args.step)))]
    files = files[0:6]
    print('found {} files'.format(len(files)), flush=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.layer_of_interest == -1:
        for i in reversed(range(4)):
            analyze_word_from_layers(args, tokenizer, files, topk_classes, i, vg_categories, model_name,
                                     args.num_t, args.num_v, args.num_tag)
    else:
        analyze_word_from_layers(args, tokenizer, files, topk_classes, args.layer_of_interest, vg_categories,
                                 model_name, args.num_t, args.num_v)


if __name__ == "__main__":
    main()
