import os
import argparse
import numpy as np
import pickle
import json
import glob
import matplotlib.pyplot as plt

from pytorch_transformers.tokenization_bert import BertTokenizer
from sklearn.manifold import TSNE
#import umap

stopwords = [line.strip('\n') for line in open('stopwords.txt')]
GLOBAL_EMBEDDING_V_LABEL = '[IMG]'
GLOBAL_EMBEDDING_T_LABEL = '[CLS]'
MASK_TOKEN_T = 103
RND_TOKEN_T = 99999

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

    parser.add_argument("--topk_t", default=100, type=int)
    parser.add_argument("--topk_v", default=25, type=int)
    parser.add_argument("--num_t", default=-1, type=int)
    parser.add_argument("--num_v", default=-1, type=int)
    parser.add_argument("--layer_of_interest", default=3, type=int)

    return parser.parse_args()

def top_visual_classes(args, vg_categories, topk=50):
    obj_stats_filepath = os.path.join(args.pkl_dir, 'sorted_obj_counts.pkl')
    obj_stats = pickle.load(open(obj_stats_filepath, 'rb'))

    topk_classes = list(obj_stats.keys())[:topk]
    for key in topk_classes:
        print('{:>8} ({:>5}) {:>8}'.format(vg_categories[key], key, obj_stats[key]), flush=True)

    return topk_classes

# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, labels, domain_labels, mask_labels, discriminator_scores, topk_t=25, topk_v=25, title=None, layer_of_interest=12, model_name='knnkl',
                   num_v=-1, num_t=-1):

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    X_v = X[domain_labels == 0]
    X_t = X[domain_labels == 1]
    labels_v = labels[domain_labels == 0]
    labels_t = labels[domain_labels == 1]
    mask_labels_v = mask_labels[domain_labels == 0]
    mask_labels_t = mask_labels[domain_labels == 1]

    if discriminator_scores.shape[0] > 0:
        has_discriminator_scores = True

    if has_discriminator_scores:
        discriminator_scores_v = discriminator_scores[domain_labels == 0]
        discriminator_scores_t = discriminator_scores[domain_labels == 1]

    def get_topk_labels(labels, topk, always_include_CLS_IMG=True):
        unique, counts = np.unique(labels, return_counts=True)
        counts = dict(zip(unique, counts))
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])}
        topk_list = list(counts.keys())[0:topk]

        if always_include_CLS_IMG:
            if '[IMG]' in counts and '[IMG]' not in topk_list:
                topk_list.append('[IMG]')
            if '[CLS]' in counts and '[CLS]' not in topk_list:
                topk_list.append('[CLS]')

        return topk_list

    topk_labels_v = get_topk_labels(labels_v, topk=topk_v)
    topk_labels_t = get_topk_labels(labels_t, topk=topk_t)

    print('top {} visual labels: {}'.format(topk_v, topk_labels_v))
    print('top {} textual labels: {}'.format(topk_t, topk_labels_t))

    selected_labels_v = np.array([True if l in topk_labels_v else False for l in labels_v])
    selected_labels_t = np.array([True if l in topk_labels_t else False for l in labels_t])

    X_v = X_v[selected_labels_v]
    X_t = X_t[selected_labels_t]
    labels_v = labels_v[selected_labels_v]
    labels_t = labels_t[selected_labels_t]
    mask_labels_v = mask_labels_v[selected_labels_v]
    mask_labels_t = mask_labels_t[selected_labels_t]

    if has_discriminator_scores:
        discriminator_scores_v = discriminator_scores_v[selected_labels_v]
        discriminator_scores_t = discriminator_scores_t[selected_labels_t]

    if num_v > 0:
        num_v = min(num_v, selected_labels_v.shape[0])
        selected_ind = np.random.choice(np.arange(labels_v.shape[0]), num_v)
        X_v = X_v[selected_ind]
        labels_v = labels_v[selected_ind]
        mask_labels_v = mask_labels_v[selected_ind]
        if has_discriminator_scores:
            discriminator_scores_v = discriminator_scores_v[selected_ind]

    if num_t > 0:
        num_t = min(num_t, selected_labels_t.shape[0])
        selected_ind = np.random.choice(np.arange(labels_t.shape[0]), num_t)
        X_t = X_t[selected_ind]
        labels_t = labels_t[selected_ind]
        mask_labels_t = mask_labels_t[selected_ind]
        if has_discriminator_scores:
            discriminator_scores_t = discriminator_scores_t[selected_ind]

    def convert(set):
        return list(set)

    # generate color map
    set_labels_v = sorted(set(labels_v))
    set_labels_t = sorted(set(labels_t))
    Y_label_color = {y: i for i, y in enumerate(convert(set_labels_v) + convert(set_labels_t))}
    Y_colors = len(Y_label_color.keys())

    Y_colors_v = len(set_labels_v)
    Y_label_color_v = {y: 0.5 * (float(i) / Y_colors_v) for i, y in enumerate(convert(set_labels_v))}

    Y_colors_t = len(set_labels_t)
    Y_label_color_t = {y: 0.5 + 0.5 * (float(i) / Y_colors_t) for i, y in enumerate(convert(set_labels_t))}

    plt.figure(figsize=(18, 9))

    # first plot: tokens / visual classes and their 2D embeddings
    ax = plt.subplot(121)
    #fontdict = {'weight': 'bold', 'size': 9}
    fontdict = {'size': 9}

    min_x = min(np.min(X_v[:, 0]), np.min(X_t[:, 0])) - 0.025
    max_x = max(np.max(X_v[:, 0]), np.max(X_t[:, 0])) + 0.025

    min_y = min(np.min(X_v[:, 1]), np.min(X_t[:, 1])) - 0.025
    max_y = max(np.max(X_v[:, 1]), np.max(X_t[:, 1])) + 0.025

    CM = plt.cm.Dark2
    for i in range(X_v.shape[0]):
        if labels_v[i] == '[IMG]':
            assert discriminator_scores_v[i] != -1
            if discriminator_scores_v[i] > 0.5:
                color = 'g'
            else:
                color = 'r'
        else:
            color = CM(Y_label_color_v[labels_v[i]])
        plt.text(X_v[i, 0], X_v[i, 1], str(labels_v[i]), color=color, fontdict=fontdict)
    for i in range(X_t.shape[0]):
        if labels_t[i] == '[CLS]':
            assert discriminator_scores_t[i] != -1
            if discriminator_scores_t[i] > 0.5:
                color = 'g'
            else:
                color = 'r'
        else:
            color = CM(Y_label_color_t[labels_t[i]])
        plt.text(X_t[i, 0], X_t[i, 1], str(labels_t[i]), color=color, fontdict=fontdict)

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    plt.subplot(122)
    markersize = 8
    #plt.plot(X_v[:, 0], X_v[:, 1], 'ko', label='visual', alpha=0.5, markeredgewidth=0.0, markersize=markersize)
    #plt.plot(X_t[:, 0], X_t[:, 1], 'k<', label='textual', alpha=0.5, markeredgewidth=0.0, markersize=markersize)

    for label in Y_label_color.keys():
        if label in set_labels_v:
            color = CM(Y_label_color_v[label])
            x = X_v[labels_v == label]
            mask_labels_target = mask_labels_v[labels_v == label]
            x_notmasked = x[mask_labels_target == -1]
            x_masked = x[mask_labels_target != -1]
            plt.plot(x_notmasked[:, 0], x_notmasked[:, 1], 'o', color=color, alpha=0.5,
                     fillstyle='full', markeredgecolor=color, markeredgewidth=0.0, markersize=markersize)
            plt.plot(x_masked[:, 0], x_masked[:, 1], 'o', color='k',
                     fillstyle='none', markeredgecolor='k', markersize=markersize)

            if x_masked.shape[0] > 0:
                template_v_masked = x_masked[0]
            if x_notmasked.shape[0] > 0:
                template_v_notmasked = x_notmasked[0]

        else:
            color = CM(Y_label_color_t[label])
            x = X_t[labels_t == label]
            mask_labels_target = mask_labels_t[labels_t == label]
            x_notmasked = x[mask_labels_target == -1]
            x_masked = x[mask_labels_target == MASK_TOKEN_T]
            x_random = x[mask_labels_target == RND_TOKEN_T]
            plt.plot(x_notmasked[:, 0], x_notmasked[:, 1], '<', color=color, alpha=0.5,
                     fillstyle='full', markeredgecolor=color, markeredgewidth=0.0, markersize=markersize)
            plt.plot(x_masked[:, 0], x_masked[:, 1], '<', color='k',
                     fillstyle='none', markeredgecolor='k', markersize=markersize)
            plt.plot(x_random[:, 0], x_random[:, 1], '<',
                     fillstyle='none', markeredgecolor='b', markersize=markersize)

            if x_masked.shape[0] > 0:
                template_t_masked = x_masked[0]
            if x_notmasked.shape[0] > 0:
                template_t_notmasked = x_notmasked[0]
            if x_random.shape[0] > 0:
                template_t_random = x_random[0]

    plt.plot(template_v_notmasked[0], template_v_notmasked[1], 'o', color='k', alpha=0.5,
             fillstyle='full', markeredgecolor='k', markeredgewidth=0.0, markersize=markersize, label='visual')
    plt.plot(template_t_notmasked[0], template_t_notmasked[1], '<', color='k', alpha=0.5,
             fillstyle='full', markeredgecolor='k', markeredgewidth=0.0, markersize=markersize, label='textual')

    plt.plot(template_v_masked[0], template_v_masked[1], 'o', color='k',
             fillstyle='none', markeredgecolor='k', markersize=markersize, label='visual (masked tokens)')
    plt.plot(template_t_masked[0], template_t_masked[1], '<', color='k',
             fillstyle='none', markeredgecolor='k', markersize=markersize, label='textual [MASK]')

    plt.plot(template_t_random[0], template_t_random[1], '<',
             fillstyle='none', markeredgecolor='b', markersize=markersize, label='textual (random tokens)')

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if title is not None:
        #plt.title(title)
        plt.suptitle(title)

    plt.tight_layout(rect=[0, 0.015, 1, 0.985])
    plt.savefig('{}_analyze_word_from_layer_{}.png'.format(model_name, layer_of_interest), dpi=150)
    #plt.show()


def analyze_word_from_layers(args, tokenizer, files, topk_classes, layer_of_interest, vg_categories, model_name,
                             num_t, num_v):
    '''

    :param files:
    :param topk_classes:
    :param layer_of_interest:   0, 1, 2, 3; 3: top layer
    :return:
    '''

    embedding_filepath = '{}_analyze_word_from_layer_{}_embeddings.pkl'.format(model_name, layer_of_interest)
    tsne_embedding_filepath = '{}_analyze_word_from_layer_{}_tsne_embeddings.pkl'.format(model_name, layer_of_interest)

    if os.path.exists(embedding_filepath):
        print('found cached embeddings at [{}]'.format(embedding_filepath))
        data = pickle.load(open(embedding_filepath, 'rb'))
        embeddings = data['embeddings']
        labels = data['labels']
        domain_labels = data['domain_labels']
        mask_labels = data['mask_labels']
        discriminator_scores = data['discriminator_scores']
    else:
        embeddings_v = []
        labels_v = []
        mask_labels_v = []  # 1: masked, 0: not masked
        discriminator_scores_v = []

        embeddings_t = []
        labels_t = []
        mask_labels_t = []  # 1: masked, 0: not masked
        discriminator_scores_t = []
        print('collecting visual embeddings...')

        cat_embeddings = {}
        for file in files:
            print('file: {}'.format(file))
            data = pickle.load(open(file, 'rb'))

            if 'discriminator_score_v' in data and 'discriminator_score_t' in data:
                has_discriminator_scores = True
            else:
                has_discriminator_scores = False
            # data['obj_labels'].shape (batch, num_regions)
            # num_layers = len(data['att_masks_tx'])
            obj_labels = data['obj_labels']     # (256, 36)

            #ones = np.ones_like(obj_labels[:, 0:1]) * 1601  # treat "1601" as [IMG] token
            #obj_labels = np.concatenate([ones, obj_labels], axis=1)

            global_embeddings_v = data['sequence_output_vx'][layer_of_interest][:, 0]   # (batch, dim)
            local_embeddings_v = data['sequence_output_vx'][layer_of_interest][:, 1::]  # (batch, n_tokens, dim)

            embeddings_v.append(global_embeddings_v)
            labels_v.append([GLOBAL_EMBEDDING_V_LABEL] * global_embeddings_v.shape[0])
            mask_labels_v.append([-1] * global_embeddings_v.shape[0])

            if has_discriminator_scores:
                discriminator_scores_v.append(data['discriminator_score_v'])

            for cat in topk_classes:
                embeddings_v_target_class = local_embeddings_v[obj_labels == cat]
                mask_v_target_class = data['image_label'][obj_labels == cat]
                assert mask_v_target_class.shape[0] == embeddings_v_target_class.shape[0]

                if embeddings_v_target_class.shape[0] > 0:
                    if cat in cat_embeddings:
                        cat_embeddings[cat]['embeddings'].append(embeddings_v_target_class)
                        cat_embeddings[cat]['mask_labels'].append(mask_v_target_class)
                    else:
                        cat_embeddings[cat] = {'embeddings': [embeddings_v_target_class],
                                               'mask_labels': [mask_v_target_class]}

        for cat in cat_embeddings:
            for key in cat_embeddings[cat]:
                cat_embeddings[cat][key] = np.concatenate(cat_embeddings[cat][key])     # (n, dim)
            #cat_embeddings[cat]['embeddings'] = np.concatenate(cat_embeddings[cat]['embeddings'])     # (n, dim)
            #cat_embeddings[cat]['mask_labels'] = np.concatenate(cat_embeddings[cat]['mask_labels'])  # (n, dim)

            embeddings_v.append(cat_embeddings[cat]['embeddings'])
            labels_v.append([vg_categories[cat]] * cat_embeddings[cat]['embeddings'].shape[0])
            mask_labels_v.append(cat_embeddings[cat]['mask_labels'])
            if has_discriminator_scores:
                discriminator_scores_v.append(-np.ones_like(cat_embeddings[cat]['mask_labels'], dtype=np.int8))

        embeddings_v = np.concatenate(embeddings_v)
        labels_v = np.concatenate(labels_v)
        mask_labels_v = np.concatenate(mask_labels_v)
        if has_discriminator_scores:
            discriminator_scores_v = np.concatenate(discriminator_scores_v)
        domain_v = np.zeros_like(labels_v, dtype=np.int8)

        print('collecting language embeddings...')
        for file in files:
            print('file: {}'.format(file))
            data = pickle.load(open(file, 'rb'))
            #global_embeddings_t = data['sequence_output_tx'][layer_of_interest][:, 0]
            #local_embeddings_t = data['sequence_output_tx'][layer_of_interest][:, 1:-1]
            embeddings = data['sequence_output_tx'][layer_of_interest]

            masked_obj_labels = data['lm_label_ids']    # (batch, n_tokens including [CLS] and [SEP])
            tokens_t = data['input_ids']                # (batch, n_tokens including [CLS] and [SEP])
            #all_tokens = tokenizer.convert_ids_to_tokens(tokens_t)
            all_tokens = [tokenizer.convert_ids_to_tokens(tokens) for tokens in tokens_t]

            # remove token that represents a stop word
            for i, tokens in enumerate(all_tokens):
                if has_discriminator_scores:
                    discriminator_scores_t.append(data['discriminator_score_t'][0])     # disc score for [CLS] token
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

                    if has_discriminator_scores:
                        if token != '[CLS]':
                            discriminator_scores_t.append(-1)

        embeddings_t = np.array(embeddings_t)
        labels_t = np.array(labels_t)
        mask_labels_t = np.array(mask_labels_t)
        if has_discriminator_scores:
            discriminator_scores_t = np.array(discriminator_scores_t)
        domain_t = np.ones_like(labels_t, dtype=np.int8)

        embeddings = np.concatenate([embeddings_t, embeddings_v])
        labels = np.concatenate([labels_t, labels_v])
        domain_labels = np.concatenate([domain_t, domain_v])
        mask_labels = np.concatenate([mask_labels_t, mask_labels_v])
        if has_discriminator_scores:
            discriminator_scores = np.concatenate([discriminator_scores_t, discriminator_scores_v])
        else:
            discriminator_scores = np.array([])

        with open(embedding_filepath, 'wb') as fp:
            pickle.dump({'embeddings': embeddings, 'labels': labels, 'domain_labels': domain_labels,
                         'mask_labels': mask_labels, 'discriminator_scores': discriminator_scores}, fp)

    print('embeddings {} labels {} domain_labels {} mask_labels {} '
          'discriminator_scores {}'.format(embeddings.shape, labels.shape, domain_labels.shape,
                                           mask_labels.shape, discriminator_scores.shape))

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
    plot_embedding(tsne_embeddings, labels, domain_labels, mask_labels, discriminator_scores, title='Model: {}, vision & language t-SNE embeddings, layer {}'.format(model_name, layers[layer_of_interest] + 1),
                   topk_t=args.topk_t, topk_v=args.topk_v, layer_of_interest=layers[layer_of_interest] + 1, model_name=model_name, num_t=num_t, num_v=num_v)

def analyze_sentence_from_layer(files):
    pass

def main():
    args = parse_args()
    model_name = os.path.basename(args.pkl_dir)
    vg_categories = json.load(open(args.vg_categories, 'rb'))['categories']
    vg_categories = {cat['id']: cat['name'] for cat in vg_categories}

    topk_classes = top_visual_classes(args, vg_categories, topk=args.topk_v)
    files = [file for file in glob.glob(os.path.join(args.pkl_dir, '0*.pkl'))]
    files = sorted(files)
    print('found {} files'.format(len(files)), flush=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.layer_of_interest == -1:
        for i in range(4):
            analyze_word_from_layers(args, tokenizer, files, topk_classes, i, vg_categories, model_name,
                                     args.num_t, args.num_v)
    else:
        analyze_word_from_layers(args, tokenizer, files, topk_classes, args.layer_of_interest, vg_categories,
                                 model_name, args.num_t, args.num_v)


if __name__ == "__main__":
    main()
