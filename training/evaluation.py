import torch
from losses import loss_fn, sequence_loss_fn


def eval_step(model, test_data, test_labels, training_target='last_token', eval_batch_size=4096):
    """Perform one evaluation step."""
    model.eval()
    num_samples = test_data.size(0)
    if num_samples == 0:
        return 0.0, 0.0, (0.0 if training_target == 'seq_cot' else None)

    total_loss_weighted = 0.0
    total_final_correct = 0
    total_final_count = 0
    total_prefix_correct = 0
    total_prefix_count = 0

    with torch.inference_mode():
        for start in range(0, num_samples, eval_batch_size):
            end = min(start + eval_batch_size, num_samples)
            batch_data = test_data[start:end]
            batch_labels = test_labels[start:end]

            if training_target == 'seq_cot':
                test_logits = model(batch_data, return_sequence_logits=True)
                batch_loss = sequence_loss_fn(test_logits, batch_labels)
                total_loss_weighted += batch_loss.item() * batch_data.size(0)

                predictions = test_logits.argmax(dim=-1)
                batch_final_labels = batch_labels[:, -1]
                batch_final_mask = batch_final_labels >= 0
                total_final_count += int(batch_final_mask.sum().item())
                total_final_correct += (predictions[:, -1][batch_final_mask] == batch_final_labels[batch_final_mask]).sum().item()

                batch_prefix_mask = batch_labels >= 0
                batch_prefix_mask[:, -1] = False
                total_prefix_count += int(batch_prefix_mask.sum().item())
                total_prefix_correct += (predictions[batch_prefix_mask] == batch_labels[batch_prefix_mask]).sum().item()
            else:
                test_logits = model(batch_data)
                batch_loss = loss_fn(test_logits, batch_labels)
                total_loss_weighted += batch_loss.item() * batch_data.size(0)

                predictions = test_logits.argmax(dim=-1)
                total_final_correct += (predictions == batch_labels).sum().item()
                total_final_count += int(batch_labels.numel())

    mean_loss = total_loss_weighted / num_samples
    test_accuracy = (total_final_correct / total_final_count * 100.0) if total_final_count else 0.0
    if training_target == 'seq_cot':
        test_prefix_accuracy = (total_prefix_correct / total_prefix_count * 100.0) if total_prefix_count else 0.0
    else:
        test_prefix_accuracy = None
    return mean_loss, test_accuracy, test_prefix_accuracy


def eval_step_multi_length(model, test_splits_by_nterms, training_target='last_token', eval_batch_size=4096):
    """Evaluate exact variable-length splits and aggregate metrics."""
    model.eval()

    total_loss_weighted = 0.0
    total_correct = 0
    total_count = 0
    total_prefix_correct = 0
    total_prefix_count = 0

    with torch.inference_mode():
        for nterms in sorted(test_splits_by_nterms.keys()):
            test_data, test_labels = test_splits_by_nterms[nterms]
            if test_data.numel() == 0:
                continue

            for start in range(0, test_data.size(0), eval_batch_size):
                end = min(start + eval_batch_size, test_data.size(0))
                batch_data = test_data[start:end]
                batch_labels = test_labels[start:end]
                if training_target == 'seq_cot':
                    logits = model(batch_data, return_sequence_logits=True)
                    loss = sequence_loss_fn(logits, batch_labels)
                    total_loss_weighted += loss.item() * batch_data.size(0)

                    predictions = logits.argmax(dim=-1)
                    final_labels = batch_labels[:, -1]
                    final_mask = final_labels >= 0
                    total_correct += (predictions[:, -1][final_mask] == final_labels[final_mask]).sum().item()
                    total_count += int(final_mask.sum().item())

                    prefix_mask = batch_labels >= 0
                    prefix_mask[:, -1] = False
                    total_prefix_correct += (predictions[prefix_mask] == batch_labels[prefix_mask]).sum().item()
                    total_prefix_count += int(prefix_mask.sum().item())
                else:
                    logits = model(batch_data)
                    loss = loss_fn(logits, batch_labels)
                    total_loss_weighted += loss.item() * batch_data.size(0)

                    predictions = logits.argmax(dim=-1)
                    total_correct += (predictions == batch_labels).sum().item()
                    total_count += int(batch_labels.numel())

    if total_count == 0:
        return 0.0, 0.0, (0.0 if training_target == 'seq_cot' else None)

    mean_loss = total_loss_weighted / total_count
    accuracy = total_correct / total_count * 100.0
    if training_target == 'seq_cot':
        prefix_accuracy = (total_prefix_correct / total_prefix_count * 100.0) if total_prefix_count else 0.0
        return mean_loss, accuracy, prefix_accuracy
    return mean_loss, accuracy, None
