# Simple Classical Waltz — C major I-vi-IV-V loop (3/4, oom-pah-pah)
# Sonic Pi 5.x — .rb — @ 88 BPM waltz — Left Hand sustained pedal / Right Hand sings
# Left: soft sustained pedal (bass 0.28, chords 0.18/0.16) — washes under
# Right: LOUD cantabile piano (0.74-0.86) — 4-bar phrase-planned, coherent:
#   nearest chord-tone voice leading per phrase, ~1 rest per 4 bars (multi-bar breath),
#   1-in-3 lead-in on beat 3 into next bar's harmony. Always chord-tones.
# Idioms: chord_cycle.rb (chord) + ambient_darin.rb phrasing + haunted.rb (one_in/choose)
# Validate: ruby -c waltz.rb

use_debug false
use_bpm 88

with_fx :reverb, room: 0.86, mix: 0.44 do
  with_fx :lpf, cutoff: 94 do

    # 8-bar loop — C  Am  F  G | C  Am  F  G  (I-vi-IV-V, Viennese waltz)
    chords = [
      chord(:C4, :major),  # I   C E G
      chord(:A3, :minor),  # vi  A C E
      chord(:F3, :major),  # IV  F A C
      chord(:G3, :major),  # V   G B D
      chord(:C4, :major),  # I
      chord(:A3, :minor),  # vi
      chord(:F3, :major),  # IV
      chord(:G3, :major),  # V
    ]
    bass = [:C2, :A1, :F2, :G2, :C2, :A1, :F2, :G2]

    # right-hand: planned 4 bars at a time for coherence (phrase-level voice leading)

    live_loop :waltz do
      # two coherent 4-bar phrases per 8-bar cycle (more musical than per-bar dice)
      2.times do |phrase|
        base = phrase * 4
        pcs = chords[base, 4]

        # -- plan 4-bar phrase: nearest chord tone / common-tone stepwise --
        phrase_notes = []
        n0 = pcs[0].choose
        n0 += 12 if n0 < note(:A4)
        phrase_notes << n0
        3.times do |j|
          prev = phrase_notes.last
          if prev.nil?
            n = pcs[j+1].choose
            n += 12 if n < note(:A4)
            phrase_notes << n
            next
          end
          if one_in(6)  # ~1 rest per phrase, chains to multi-bar breath across phrases
            phrase_notes << nil
            next
          end
          # build candidates in G3-C6 range without select/sort_by (Sonic Pi spider safe)
          cands = []
          pcs[j+1].each do |p|
            [p, p+12, p-12].each do |q|
              if q >= note(:G3) && q <= note(:C6)
                cands << q
              end
            end
          end
          # nearest stepwise / common-tone — linear scan, no sort_by
          best = cands[0]
          best_d = (best - prev).abs
          cands.each do |q|
            d = (q - prev).abs
            if d < best_d
              best = q
              best_d = d
            end
          end
          # allow slight variation: 1-in-3 pick second-nearest instead of best
          if one_in(3)
            # find second nearest
            second = nil
            second_d = 1000
            cands.each do |q|
              next if q == best
              d = (q - prev).abs
              if d < second_d
                second = q
                second_d = d
              end
            end
            best = second if !second.nil? && one_in(2)
          end
          phrase_notes << best
        end

        4.times do |j|
          i = base + j
          c = chords[i]
          b = bass[i]
          n = phrase_notes[j]
          nxt = (j < 3) ? phrase_notes[j+1] : nil

          # -- RIGHT HAND : phrase-coherent — rests & lead-ins stay inside phrase --
          in_thread do
            use_synth :piano
            if n.nil?
              sleep 3
            else
              amp = rrand(0.74, 0.86)
              if !nxt.nil? && one_in(3)
                play n, attack: 0.5, sustain: 0.85, release: 1.1, amp: amp, cutoff: 92
                sleep 2
                play nxt, attack: 0.16, sustain: 0.28, release: 0.85, amp: amp*0.92, cutoff: 92
                sleep 1
              else
                play n, attack: 0.5, sustain: 1.55, release: 1.5, amp: amp, cutoff: 92
                sleep 3
              end
            end
          end

          # -- LEFT HAND : sustained pedal — softer, background wash --
          use_synth :piano
          play b, attack: 0.07, sustain: 2.4, release: 1.1, amp: 0.28, cutoff: 76
          sleep 1
          play_chord c, attack: 0.07, sustain: 0.85, release: 1.35, amp: 0.18, cutoff: 82
          sleep 1
          play_chord c, attack: 0.07, sustain: 0.85, release: 1.35, amp: 0.16, cutoff: 82
          sleep 1
        end
      end
    end

  end
end
