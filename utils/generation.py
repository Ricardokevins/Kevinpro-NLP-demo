from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

tokenizer = AutoTokenizer.from_pretrained("/Users/sheshuaijie/Desktop/RearchSpace/Data/PLM/linydub-bart-large-samsum")
model = AutoModelForSeq2SeqLM.from_pretrained("/Users/sheshuaijie/Desktop/RearchSpace/Data/PLM/linydub-bart-large-samsum")
input_prompt = "Aude: Hi Susie, how is Ted this morning? Did you find plasters?\nSusie: yes. He kept them till this morning after his shower.\nAude: he must look sexy whith them... lol\nSusie: a  memory from  Poland!"
inputs = tokenizer([input_prompt], max_length=1024, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], num_beams=1, min_length=0, max_length=30)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])
exit()
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

# instantiate logits processors
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(10, eos_token_id=tokenizer.eos_token_id),
    ]
)
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

outputs = model.greedy_search(
    input_ids = input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
)

print(tokenizer.batch_decode(outputs, skip_special_tokens=False))


def _update_model_kwargs_for_generation(
    outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
) -> Dict[str, Any]:
    # update past
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs.past_key_values
    elif "mems" in outputs:
        model_kwargs["past"] = outputs.mems
    elif "past_buckets_states" in outputs:
        model_kwargs["past"] = outputs.past_buckets_states
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return model_kwargs

def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)
            # for name in model_inputs:
            #     print(name)
            # print(model_inputs)
            # exit()

            print(model_inputs['decoder_input_ids'])
            #print(model_inputs)
            #exit()
            # forward pass to get next token
            
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            #BOOKMARK


            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # model_kwargs = _update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=True
            # )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id is not None:
            #     unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # # stop when each sentence is finished, or if we exceed the maximum length
            # if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            #     if not synced_gpus:
            #         break
            #     else:
            #         this_peer_finished = True

        return input_ids