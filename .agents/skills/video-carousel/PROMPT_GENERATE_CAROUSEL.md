You are generating video carousels in the style of `thefallenpoet` — melancholic, philosophical, noir.

**Reference:** The JSON below is `thefallenpoet/keyframes-flattened/combined.json` — list-of-lists: 6 exemplar carousels, 32 frames total, deduped (consecutive duplicate `text` removed). Each exemplar is `Array<Frame>`. Study it as few-shot before generating.

**Task:** Create 20+ NEW carousel ideas about **{{TOPIC}}**. Do not copy text or visuals verbatim — match the DNA, not the content. Replace `{{TOPIC}}` with your subject (e.g. "healing after loss", "stoic resilience", "quiet ambition") — keep the noir style even when the topic is soft.

Each carousel is `Array<Frame>` (3-7 frames, ideal 5):
```json
{
  "text": "short overlay text — 1 line, poetic/philosophical hook",
  "visual_description": "vivid genAI image prompt — composition + subject + lighting + palette + mood + style tags"
}
```

**Style DNA (from combined.json + Retro Avant-Garde Noir):**
- Text: intimate 2nd-person or quoted philosopher, line-broken for pacing (hook → build → payoff). Lowercase where intentional, one idea per frame. No hashtags.
- Visual: `Retro Avant-Garde Noir — Atmospheric & Moody` — 100% hand-painted (not CGI/photo), painterly digital illustration / graphic novel / 90s cel-shade, visible brushwork + ink outlines, film-noir chiaroscuro, deep blacks + warm amber/burnt-orange glow through crimson/maroon, muted earth tones with desaturated blues/teals, vertical portrait framing, medium/close-up, solitary or distanced pair, pensive/melancholic mood.

**Visual description rules:** Must be usable as Flux/SD prompt. Always include: shot type, subject pose/clothing, background, lighting, color palette, mood, style keywords. Vary compositions across a carousel but keep palette/style coherent. One distinct image per frame.

**Output:** Single JSON file — list of 20+ carousels:
```json
[
  [ {"text": "...", "visual_description": "..."}, ... ],
  [ {"text": "...", "visual_description": "..."}, ... ]
]
```
No prose outside JSON, no markdown wrapping. Validate JSON.

**Quality check:** No consecutive duplicate `text` within a carousel, no repeated visual_description, each carousel tells a complete micro-narrative.

---

## REFERENCE_JSON — `combined.json` (6 carousels, 32 frames)

```json
[
  [
    {
      "text": "How true it was when Albert Camus said:",
      "visual_description": "A young man with short dark hair sits at a table, gazing upward with a pensive, melancholic expression. He wears a beige trench coat over a blue collared shirt and dark tie. A cigarette is in his mouth, emitting a thin wisp of smoke. His hands rest on scattered papers. Dramatic noir lighting casts a warm, reddish glow on the left side of his face and coat, contrasting with deep shadows on the right. The background is a dark void with faint white specks resembling stars. The image is a painterly digital illustration with muted earth tones, capturing a moody, introspective atmosphere in a medium close-up shot."
    },
    {
      "text": "\"If you want to know a person,",
      "visual_description": "A medium close-up illustration depicts a man with dark hair and a thick mustache, looking upward with an intense, wide-eyed expression. He wears a textured, maroon-brown jacket over a dark shirt, his right hand raised to touch his forehead in a gesture of distress or deep thought. The background is a dark teal-green featuring vertical streaks of white and pink, resembling rain or abstract brushstrokes, with a solid red strip on the right edge. The art style is sketchy and illustrative, utilizing cross-hatching textures and muted, earthy tones to create a melancholic, noir-like mood. The lighting highlights his face and hand against the darker, patterned backdrop."
    },
    {
      "text": "discuss politics,",
      "visual_description": "Two men stand side-by-side in a vertical medium shot, rendered in a graphic novel noir style with bold outlines and flat shading. The figure on the left wears a tan trench coat with black buttons, a dark tie, and a brown fedora, his hands tucked into his pockets. Beside him, the man on the right wears a dark maroon suit jacket, white shirt, and matching fedora, also with hands in pockets. Both characters have serious, stoic expressions with shadowed faces. The background features a cloudy sky in shades of purple and grey above a faint, indistinct horizon. The color palette is muted and earthy, dominated by browns, tans, and maroons, creating a somber and mysterious mood."
    },
    {
      "text": "religion,",
      "visual_description": "A horizontal illustration depicts five men seated in a row, engaged in a serious discussion. They appear to be historical figures, dressed in 19th-century attire. The man on the far left sits on a red chair, wearing a white robe with legs crossed. Next to him, a man with a long white beard wears a dark suit. The central figure gestures with his hands while wearing a dark jacket. To his right, a bearded man in light grey clothing listens intently, and a fifth figure is partially visible on the far right. The background is a dark, textured grey wall with a reddish-brown upper border. The lighting is dim and moody, casting soft shadows. The color palette is muted, featuring deep reds, browns, and greys. The style is painterly and sketchy, evoking a vintage, somber mood."
    },
    {
      "text": "and women with him.\"",
      "visual_description": "Two young women lean out of a window at night, framed by a dark red wooden shutter on the left. The woman on the left wears a white peasant blouse with a blue bodice and offers a subtle, calm smile. Beside her, the second woman, dressed in a similar white blouse with puffy sleeves, rests her chin on her hand with a wide-eyed, excited expression, as if sharing a secret. The background is a deep, pitch-black void, emphasizing the intimacy of the moment. Soft, warm light illuminates their faces and clothing from the left, creating a dramatic contrast against the darkness. The illustration style is clean and graphic, evoking a vintage, storybook mood."
    }
  ],
  [
    {
      "text": "Just because someone is keeping you around,",
      "visual_description": "A digital illustration depicts a tense scene with a man and a woman sitting on a long, mauve couch against a deep crimson wall. The man, on the left, wears a dark suit and white shirt, reclining with his arm draped over the backrest, looking toward the woman with a serious expression. The woman, on the right, wears a dark, sleeveless backless dress and sits facing him but turned slightly away. The background is a gradient of intense red fading to black at the top. The lighting is dim and moody, emphasizing the emotional distance between the figures. The style is illustrative and noir-like, with flat shading and bold colors."
    },
    {
      "text": "doesn't mean they are choosing you.",
      "visual_description": "A digital illustration shows a man and a woman sitting on a pink sofa in a room saturated with deep red, moody lighting. The man, on the left, wears a black suit and tie, reclining with one arm extended along the back of the couch, his expression pensive. The woman, on the right, wears a dark, sleeveless dress with a floral pattern, sitting upright with her back mostly turned toward him. The background is dark and shadowy, enhancing the somber and tense atmosphere. The composition places the figures on opposite ends of the couch, suggesting emotional distance. The art style is illustrative with flat shading, resembling a graphic novel panel."
    },
    {
      "text": "Some people will enjoy your presence,",
      "visual_description": "A vertical digital illustration depicts a Regency-era scene defined by dramatic, high-contrast lighting. On the left, a man stands in silhouette with his back to the viewer, wearing a black top hat and a long, dark coat. Opposite him, a woman stands in a pale, empire-waist ballgown with puffed sleeves, holding a vibrant red parasol. She looks back over her shoulder with a soft, contemplative expression. A sharp beam of light cuts diagonally from the top right, illuminating the woman and the textured wall behind her, while the man remains largely in shadow. The palette features muted blues and blacks contrasting with the soft white dress, creating a romantic and mysterious mood."
    },
    {
      "text": "your attention,",
      "visual_description": "A digital illustration shows a man and woman sitting closely on brick steps outside a dark building at night. The man, dressed in a dark suit jacket, tie, and patterned trousers, sits slightly behind the woman, his arm resting around her shoulders. The woman, with short blonde hair, wears a white sleeveless dress with a flowing, patterned skirt. They gaze at each other with serious, intimate expressions. Warm, amber light filters through a tall, multi-paned window on the left, illuminating their faces and casting deep shadows against the teal-blue wall. The scene is moody and romantic, rendered in a painterly style with a palette of deep blues, greens, and warm highlights. Vertical composition focuses on their connection."
    },
    {
      "text": "and the comfort of knowing you're there,",
      "visual_description": "A digital illustration depicts a man and a woman sitting intimately on brick steps at night. The man, wearing a dark suit jacket, tie, and light trousers, sits behind the woman with his arm draped protectively over her shoulder. The woman, with blonde hair, wears a white patterned sleeveless dress and leans back against him, gazing upward with a calm, contemplative expression. Warm, golden light streams from a window in the upper left, illuminating the man's face slightly and casting long shadows across the dark blue and black surroundings. The scene is moody and cinematic, with deep shadows contrasting against the soft glow of the window and the white dress. The composition is vertical, focusing on the couple's connection against a dark architectural background."
    },
    {
      "text": "but they'll never build a future with you.",
      "visual_description": "An illustrative digital painting depicts a man and a woman sitting back-to-back on a grassy bank. The woman, on the left, wears a light blue dress and faces left, while the man, on the right, wears a white t-shirt and faces right. Both are shown in profile with neutral, contemplative expressions. In the immediate foreground, dark red flowers dot the grass. Behind the couple lies a calm blue lake stretching toward a dark, silhouetted hill under a pale, cloudy sky. The color palette is muted and cool, dominated by blues, whites, and deep greens, evoking a quiet, melancholic mood with a textured, painterly style."
    },
    {
      "text": "Don't mistake being an option",
      "visual_description": "Two figures stand on a wooden dock facing each other, separated by space. A woman in a long dark coat leans against a wooden piling on the left. A man in a dark suit with arms crossed leans against a piling on the right. Both have somber, serious expressions. Behind them lies a vast blue lake stretching to hazy blue mountains under a cloudy, twilight sky. The scene is framed by the two vertical wooden posts. The lighting is soft and cool, dominated by shades of blue, black, and brown. The style is illustrative and painterly, evoking a moody, melancholic atmosphere. The composition centers the figures, emphasizing their isolation against the landscape."
    },
    {
      "text": "for being a priority.",
      "visual_description": "An illustrative painting depicts a man and a woman standing on a wooden dock, separated by a vertical wooden piling. Both figures face each other in profile, locked in a somber gaze. The woman, on the left, wears a long dark coat and heels. The man, on the right, wears a dark suit. The background features a calm blue lake stretching toward distant, hazy mountains and pine forests under a cloudy, muted sky. The lighting is soft and diffuse, evoking a melancholic, overcast atmosphere. The color palette is dominated by cool blues, deep greens, and dark earth tones. The composition is centered, with the figures framed by the wooden pilings, emphasizing a sense of distance. The style is painterly and slightly vintage."
    },
    {
      "text": "The right person won't make you wonder where you stand.",
      "visual_description": "A digital illustration depicts a couple embracing on a city street at night. The man, seen from behind, wears a blue jacket featuring a prominent golden dragon embroidery on the back and dark trousers. He holds a woman close; she wears a long, flowing red dress or coat and carries a white shopping bag. Her face is hidden against his shoulder, emphasizing intimacy. The scene is bathed in cool blue and teal ambient light, contrasting with the warm yellow glow emanating from a storefront window behind them. Silhouetted figures of passersby flank the couple on the left and right, adding depth to the urban setting. The style is illustrative with a soft, slightly grainy texture, capturing a quiet, romantic moment amidst the bustle of the city."
    },
    {
      "text": "They'll make sure you never have to.",
      "visual_description": "A digital illustration depicts a couple embracing from behind on a city street at night. The man, with dark hair, wears a blue jacket featuring a prominent golden dragon design on the back and dark trousers. The woman, with long dark hair, wears a long red dress and holds a white tote bag. They stand in front of a doorway with blue framing and a red sign above. The scene is bathed in cool, cinematic blue lighting, contrasting with the woman's red dress. Silhouettes of bystanders flank the couple on the left and right. The style is reminiscent of anime or modern digital painting, with a moody, romantic atmosphere. The camera framing is a medium shot, capturing the couple from the thighs up."
    }
  ],
  [
    {
      "text": "How true it was when Seneca asked,",
      "visual_description": "A digital painting depicts an elderly, wise-looking man with curly gray hair and a full beard, resembling an ancient Greek philosopher. He wears a textured white tunic with a teal trim. His head is tilted upward, gazing intently towards the upper right with a contemplative, questioning expression. Dramatic lighting illuminates his forehead, nose, and beard from the right, casting the left side of his face and neck into deep shadow against a solid black background. The style is illustrative with warm skin tones and high contrast. The composition is a close-up bust shot, focusing entirely on his expressive face and upper shoulders."
    },
    {
      "text": "\"The final test of freedom is this:",
      "visual_description": "A digital illustration depicts two men in profile facing right against a textured, deep red background. The foreground figure wears a beige draped toga and a green laurel wreath, his expression stoic and serious. His bare, muscular left arm reaches out to rest a hand on the shoulder of the second figure, who stands slightly behind wearing a dark red hooded garment. The lighting is dramatic, casting shadows across their faces while highlighting their profiles. The style is painterly and gritty, evoking a historical or classical atmosphere. The composition is a close-up, emphasizing the solemn connection between the figures."
    },
    {
      "text": "Can you remain the same person in praise,",
      "visual_description": "A side-profile illustration depicts a person in a loose, dusty-brown robe holding a white theatrical mask up to their face. The figure's mouth is open in an expression of intensity or speech, while the mask, held by a hand near the eye, remains blank and pale. The lighting is dramatic and moody, casting strong highlights on the mask and the person's arm while leaving the right side and background in deep shadow. The background suggests a dark wooden door or paneling. The color palette is dominated by sepia, beige, and dark browns, creating a somber, introspective mood. The style is painterly with visible brushstrokes and textured shading."
    },
    {
      "text": "in insult, in gain and in loss?\"",
      "visual_description": "A digital illustration depicts a melancholic male jester slumped in a seated pose. He wears a vibrant, deep red jester costume with jagged white stitching details along the collar and sleeves. His matching red cap features three points, each tipped with a silver bell. His head is bowed low, eyes cast downward in an expression of profound weariness and sadness. His hands are clasped loosely in his lap. The lighting is dramatic and moody, casting deep shadows across his pale face and the folds of his red suit, while the background is a stark, pitch-black void that isolates the figure. The art style resembles a gritty comic book panel with high contrast and rich, saturated reds against the darkness."
    }
  ],
  [
    {
      "text": "Wanna block me, then block me.",
      "visual_description": "A digital illustration depicts a solitary figure with dark, shoulder-length hair sitting alone in a dimly lit movie theater. The person, wearing a black long-sleeved top, holds a white paper cup to their lips, partially obscuring their mouth. Their eyes gaze forward with a neutral, contemplative expression. The theater is filled with rows of plush, deep crimson red seats that recede into the background, creating a strong sense of depth. The lighting is low-key and moody, casting shadows that blend the subject into the dark surroundings, with only the cup and hand illuminated. The style is textured and slightly grainy, evoking a quiet, melancholic atmosphere. The composition centers the figure amidst the empty red rows, emphasizing isolation."
    },
    {
      "text": "Wanna replace me, then replace me.",
      "visual_description": "An over-the-shoulder illustration depicts a person with dark hair wearing a black coat in the left foreground, gazing out toward a sandy beach. In the distance, two small figures stand near the water's edge; one wears a maroon coat and the other a brownish garment. The horizon divides the composition, featuring a vast, textured lavender sky above and a teal-green ocean below. The beach is rendered in muted sandy browns with faint, indistinct footprints. The visual style is grainy and textured, resembling a sketch or watercolor with a nostalgic, desaturated color palette. The mood is melancholic and reflective, evoking a quiet sense of memory. Lighting is soft and diffused, suggesting an overcast day."
    },
    {
      "text": "Wanna hurt me, then hurt me.",
      "visual_description": "An illustrative, comic-style image depicts a man and a woman sitting on outdoor steps in a somber, intimate scene. The man, dressed in a dark suit jacket, light blue tie, and white trousers, sits on the upper step, leaning forward with a downcast, contemplative gaze. Below him, a woman in a pale blue, sleeveless dress sits barefoot, her body angled toward him but her eyes looking away with a distant expression. The setting appears to be a porch at dusk or in shadow, featuring dark green siding, a dark doorway behind the man, and a window emitting a soft yellow glow behind the woman. White flowers bloom in the bottom left corner. The lighting is moody and low-key, emphasizing deep blues and shadows, creating a nostalgic and melancholic atmosphere. The composition is vertical, capturing the emotional distance between the two figures."
    },
    {
      "text": "Wanna stop talking to me, then stop.",
      "visual_description": "This digital anime illustration in a 90s cel-shaded style features a young Black male with short, dark curly hair. He is captured in a medium close-up, leaning forward with his chin resting in his cupped hands. He wears a dark blue long-sleeved shirt. His expression is contemplative and slightly melancholic, gazing slightly upward. The lighting is dramatic and high-contrast; a warm, reddish-orange light strikes the left side of his face and hands, while the right side remains in deep shadow. The background is a deep, moody black, fading into darkness. The composition centers the subject against the dark void, emphasizing the interplay of cool blue tones and warm light."
    },
    {
      "text": "Just remember that I was there for you whenever you need me.",
      "visual_description": "A person with dark hair sits in profile facing left, gazing out a window at a dramatic twilight sky. The figure is largely silhouetted in deep shadow, wearing dark clothing, with only the side of their face and hand faintly illuminated by the ambient light. The sky outside features vibrant magenta and purple clouds against a deep indigo background, suggesting dusk. The lighting is moody and atmospheric, creating a strong contrast between the dark interior and the colorful exterior. The composition places the subject in the foreground right, looking towards the open space. The art style is reminiscent of anime, with detailed cloud textures and a melancholic, nostalgic mood."
    },
    {
      "text": "I tried my best to put a smile on your face.",
      "visual_description": "A vintage-style painting depicts a man and a woman standing on a wooden pier facing each other. The woman, on the left, wears a dark coat and skirt, standing with her hands at her sides. The man, on the right, wears a dark suit with arms crossed, leaning against a thick wooden post. They gaze at one another with serious, contemplative expressions. Behind them, a calm blue lake stretches toward distant, tree-covered mountains under a cloudy, twilight sky. The scene is framed by two vertical wooden posts in the foreground. The color palette is dominated by cool blues, deep browns, and muted grays, evoking a melancholic and romantic mood. The lighting suggests dusk, with soft reflections on the water surface."
    }
  ],
  [
    {
      "text": "We live in a world where the funeral matters more than the dead.",
      "visual_description": "A digital painting depicts a somber cemetery scene at dusk. In the foreground, a woman in a long black coat and veil bends slightly, reaching toward pink and white flowers placed near a grassy grave mound, accompanied by a small child in a black coat and hat standing silently beside her. Several white candles glow warmly on the ground. In the mid-ground, other mourners in dark clothing stand among pale grey tombstones and crosses. The background features tall green trees and a house under a twilight sky transitioning from deep blue to reddish-orange. The style is illustrative and textured, evoking a mood of quiet mourning and reverence."
    },
    {
      "text": "The wedding more than love.",
      "visual_description": "A digital illustration depicts a solemn wedding ceremony inside a dimly lit church. In the foreground, a bride with dark hair adorned with a floral crown and veil wears an intricate white off-the-shoulder gown. She looks downcast, holding the hands of a groom in a black tuxedo and white bow tie, who gazes intently at their joined hands. To the left, a priest in red and gold vestments holds a tall, lit candle. The background is filled with a crowd of formally dressed guests observing the scene in shadow. The lighting is warm and dramatic, casting deep shadows and highlighting the characters' serious expressions. The style is painterly and detailed, evoking a moody, intimate atmosphere."
    },
    {
      "text": "In the physical rather than the intellect.",
      "visual_description": "A vertical digital illustration depicts a romantic, intimate scene between a woman and a man against a stark black background. The woman, positioned in the foreground, wears a white, off-the-shoulder Regency-style gown with ruffled layers. Her head is tilted back, eyes closed, and lips slightly parted in a serene expression. Behind her, a man in a red military uniform with gold epaulettes leans in close, his face near her neck as if kissing or whispering. His hand rests on her waist. The lighting is dramatic and low-key, highlighting the woman's pale skin and dress while leaving the man partially in shadow. The style is painterly with soft textures, evoking a moody, gothic romance atmosphere."
    },
    {
      "text": "We live in the container culture which despises the content.",
      "visual_description": "A close-up profile view of a young man with messy dark hair, gazing downward with a somber, contemplative expression. He holds a silver lighter in both hands, cupping the small, bright flame. The fire provides the sole illumination, casting a warm, reddish-orange glow across his pale skin, nose, and hands, while the rest of his face and dark clothing fade into deep shadow. The background is a void of pure blackness. The lighting creates high contrast, emphasizing the intimacy and melancholy of the moment. The style is a moody, atmospheric digital illustration reminiscent of dark anime or manga art, focusing on the interplay of light and shadow."
    }
  ],
  [
    {
      "text": "Your death is way more guaranteed than your wedding day.",
      "visual_description": "A stylized digital illustration features a black crow perched on the curved top of a weathered, grey gravestone in a moonlit graveyard. The bird faces right, its plumage rendered in deep blues and blacks with a subtle sheen. The gravestone stands vertically in the foreground, marked with faint, scratchy white lines. Behind it, a grassy field in muted olive green stretches toward a row of silhouetted, rounded tombstones and dark, shadowy trees against a deep teal night sky. The lighting is low and atmospheric, casting soft highlights on the bird and stone while leaving the background in shadow. The mood is somber and mysterious, presented in a flat, graphic art style with clean lines."
    },
    {
      "text": "So instead of looking for the love of your life,",
      "visual_description": "A romantic digital illustration depicts a couple sharing an intimate moment amidst ancient stone ruins at twilight. A woman with reddish-brown hair, wearing a sleeveless tan dress, sits on a large rock block, looking up adoringly. Beside her, a man with dark curly hair, dressed in a dark blue suit jacket, pink shirt, and tie, crouches down to hold her hands, gazing back at her. The background features vertical stone pillars and rocky terrain under a sky streaked with lavender, purple, and soft pink clouds. The lighting is cool and moody, dominated by blues and purples, creating a serene and tender atmosphere. The style is clean and illustrative, with a focus on emotional connection."
    },
    {
      "text": "start living your life with love.",
      "visual_description": "A digital illustration depicts a melancholic young man with messy dark hair, looking downward with a somber expression. He wears a white, high-collared Victorian-style shirt with buttons and a dark jacket. He clutches a bouquet of deep red roses to his chest with both hands, the thorny stems visible. The lighting is dim and moody, casting warm reddish-brown hues over his face and the flowers against a textured, dark sepia background. The style features sketchy, expressive line work with a gothic, romantic atmosphere. The composition is a vertical medium shot, focusing on his emotional state and the symbolic roses he holds close."
    }
  ]
]
```

*Attach `combined.json` when prompting. Fallback: embed the REFERENCE_JSON above.*
