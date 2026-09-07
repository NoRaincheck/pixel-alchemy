# Ghost in the Shell — "Making of a Cyborg" (Kawai) inspired ritual choral loop
# Sonic Pi 5.x — .rb — 48 BPM ritual procession, temple reverb, Bulgarian cluster + taiko
# Idioms: ambient_darin.rb (phasing live_loops) + choral.rb (ring + :ambi_choir) + Satie drone
# Run: open in Sonic Pi and hit Run — or: uv run python -m pixel_alchemy.incidental.cli --wav (offline render needs .rb)
# Validate: ruby -c ghost_choir.rb

use_debug false
use_bpm 48

with_fx :reverb, room: 0.92, mix: 0.65, damp: 0.85 do
  with_fx :lpf, cutoff: 98 do
    with_fx :gverb, room: 22, spread: 0.5, damp: 0.65, mix: 0.22 do

      # -- deep pedal drone : dark_ambience holds root+fifth (temple organ) --
      live_loop :drone do
        use_synth :dark_ambience
        # D pedal — Japanese hirajoshi centre; G is the fourth (as in original)
        play :D2, attack: 6, sustain: 8, release: 6, amp: 0.36, cutoff: 78
        play :A2, attack: 6, sustain: 8, release: 6, amp: 0.28, cutoff: 78
        sleep 16
      end

      # -- Kawai cluster choir: 3 out-of-phase layers from ambient_darin.rb --
      # rate chosen from ring so each iteration retunes by a minor/major 2nd
      # = wedding chant voices stacked in dissonant seconds (Bulgarian/japanese folk)
      live_loop :choir_low do
        r = (ring 0.50, 0.53, 0.56).choose   # ~ D2 / Eb2 / E2 when sample is at ~ D3
        sample :ambi_choir, rate: r, amp: 0.44, attack: 1.5, release: 3, pan: rrand(-0.35, 0.35), cutoff: 95
        sleep [8, 10].choose
      end

      live_loop :choir_mid do
        r = (ring 0.50, 0.53, 0.60).choose
        sample :ambi_choir, rate: r * 2, amp: 0.30, attack: 1.2, release: 2.8, cutoff: rrand(85, 100), pan: rrand(-0.65, 0.65)
        sleep [11, 13].choose
      end

      # -- procedural upper voices: stepwise drift (hold + crotchet shift) --
      # each voice holds current chord tone, then with ~40% prob steps +-1 to adjacent node
      # drift is aimless random walk clamped to chord ring; sleep 1 = crotchet at 48 BPM
      idx_a = 2
      idx_b = 4
      upper_ring = ring(:D5, :Eb5, :G5, :A5, :Bb5)  # max 22st (+detune headroom <24)

      live_loop :choir_high_a do
        n = upper_ring[idx_a]
        # rate shift so :ambi_choir (base ~ C4/60) sings n — rate avoids :pitch <=24 limit
        sample :ambi_choir, rate: (2 ** ((note(n) - 60) / 12.0)), amp: 0.22, attack: 0.6, release: 1.1, cutoff: 92, pan: -0.35
        sleep 1  # crotchet step
        # hold 60% (step 0), drift up/down 20% each — aimless
        step = choose([0, 0, 0, -1, 1])
        idx_a = (idx_a + step) % upper_ring.length
      end

      live_loop :choir_high_b do
        n = upper_ring[idx_b]
        sample :ambi_choir, rate: (2 ** ((note(n) - 60) / 12.0)) * (2 ** (rrand(-0.08, 0.08) / 12.0)), amp: 0.18, attack: 0.8, release: 1.4, cutoff: 90, pan: 0.38
        sleep 1
        step = choose([0, 0, 0, -1, 1])
        idx_b = (idx_b + step) % upper_ring.length
        # occasional extra hold (double crotchet) to desync voices
        sleep 1 if one_in(4)
      end

      # -- elongated shrine melody: slow Japanese scale (hirajoshi) --
      # guidance: choose over chord tones + scale passing tones, sleep [4,6,8] with occasional rest
      live_loop :shrine_voice do
        use_synth :hollow
        # hirajoshi on D (D Eb G A Bb) — processional wedding mode; also :iwato available
        vox = (scale :D4, :hirajoshi) + (scale :D5, :hirajoshi)
        n = choose(vox)
        n = choose([:D4, :G4, :A4, :Bb4, :D5]) if one_in(3) == false  # bias to drone tones 70%
        play n, attack: 4, sustain: 4, release: 5, amp: 0.22, cutoff: 88
        sleep [4, 6, 8].choose
        sleep 2 if one_in(5)
      end

      # -- taiko processional: sparse, heavy, ritual — every ~8 beats like the film score --
      live_loop :taiko do
        sync :drone if one_in(6) # occasionally re-anchor to drone for temple steadiness
        sample :bd_boom, amp: 1.15, cutoff: 72, rate: 0.88, pan: rrand(-0.12, 0.12)
        sleep 4
        sample :drum_bass_hard, amp: 0.95, rate: 0.84, cutoff: 85
        sleep 3.5
        # occasional light kachi hit / rim roll — aleatoric per haunted.rb
        if one_in(3)
          sleep 0.5
          sample :perc_snap, rate: 0.55, amp: 0.32, pan: rrand(-0.4, 0.4)
        end
        sleep rrand(2.5, 4.0)
      end

      # -- sparse temple bell (haunted.rb idiom: aleatoric + rrand rate) --
      live_loop :temple_bell do
        sample :perc_bell, rate: rrand(0.35, 0.55), amp: 0.18, pan: rrand(-0.9, 0.9), cutoff: 90
        sleep rrand(6, 12)
      end

    end
  end
end
