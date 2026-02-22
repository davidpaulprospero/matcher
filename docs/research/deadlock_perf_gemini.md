# Research: Deadlock game Source 2 engine launch options best fps convars r_shadows mat_fullbright viewport_scale cl_updaterate graphics optimization guide competitive settings 2025 2026
**Backend:** gemini (gemini-2.0-flash)

Okay, I will research the topic of optimizing "Deadlock: The Online Shooter" (assuming this is the "Deadlock game" you're referring to) within the Source 2 engine, focusing on launch options, convars, and settings for achieving the best FPS and competitive performance in 2025-2026.

### Executive Summary

Optimizing "Deadlock: The Online Shooter" (developed by Valve and running on the Source 2 engine) for maximum FPS and competitive performance involves a combination of in-game settings adjustments, launch options, and system-level configurations. The Source 2 engine, known for its efficient 3D rendering and support for modern features like Vulkan and DirectX 11, still requires careful tuning to achieve the best possible experience, especially on a wide range of hardware. Key areas to focus on include reducing visual clutter, prioritizing GPU and CPU usage, and minimizing input latency.

Achieving optimal performance in "Deadlock" for competitive play in 2025-2026 necessitates a multifaceted approach. Gamers often tweak in-game graphics settings, utilize specific launch options, and adjust system configurations to maximize FPS and minimize input lag. Community-driven guides and pro player settings serve as valuable resources for tailoring the game to individual hardware and preferences. While specific "best" settings may vary, common recommendations include disabling resource-intensive visual effects, using appropriate upscaling techniques, and optimizing CPU and GPU priority.

### Key Findings with Source Attribution

*   **Graphics Settings:**
    *   **Upscaling Technology:** FSR2 (FidelityFX Super Resolution 2) or TAA (Temporal Anti-Aliasing) is generally recommended, with a scaling mode of "Performance" or "Balanced".
    *   **Ambient Occlusion:** Screen Space AO and Distance Field AO should be turned off to improve performance.
    *   **Motion Blur:** Disabling motion blur can provide a significant FPS boost. One source suggests a 16% increase in FPS by turning motion blur off.
    *   **Texture and Shadow Quality:** Setting these to "Medium" often provides a good balance between visual quality and performance.
    *   **Display Mode:** Fullscreen mode is recommended to ensure the game has primary access to PC resources.
    *   **Reduce Flashing Effects:** Turning this setting "On" can improve visibility and performance during intense gameplay.
*   **Launch Options:**
    *   Launch options can be set via Steam to modify game behavior.
    *   `-high` can be added to prioritize the game's CPU usage.
    *   `-novid` skips the intro video for faster loading.
    *   `-nojoy` disables joystick support, freeing up resources.
    *   `+fps_max 0` uncaps the framerate, though capping it to your monitor's refresh rate might be better if you experience thermal throttling.
    *   `-softparticlesdefaultoff` and `+mat_disable_fancy_blending 1` are launch options that can be used.
*   **Configuration Files:**
    *   Create an `autoexec.cfg` file to automatically execute custom commands. Add `+exec autoexec.cfg` to launch options to ensure it loads.
*   **System Tweaks:**
    *   **Nvidia Control Panel:** Set "Low Latency Mode" to "On". Setting it to "Ultra" may cause issues. Set power management mode to "Prefer Maximum Performance".
    *   **Game Mode:** Enable Game Mode in Windows settings to disable background processes.
    *   **Full Screen Optimizations:** Disable full screen optimizations in the game executable's compatibility settings.
    *   **Process Lasso:** Use Process Lasso to set CPU priority to "High" and disable SMT (Simultaneous Multithreading) for the game process. Disable efficiency mode and enable boost mode.
    *   **Refresh Rate:** Ensure your monitor's refresh rate is set to the highest value.
*   **Crosshair Settings:**
    *   While crosshair settings don't impact performance, they are important for gameplay. Recommended colors are Cyan, Green, or Yellow. Turn on Static Pip Gap and Pip Border to avoid distracting animations.

### Technical Details

*   **Source 2 Engine:** "Deadlock" utilizes Valve's Source 2 engine, which supports modern rendering techniques and 64-bit architecture for improved performance.
*   **ConVars:** These console variables allow users to fine-tune various aspects of the game, including graphics, networking, and audio. The `cvarlist` command can be used to display all available convars.
*   **DirectX and Vulkan:** The game supports both DirectX 11 and Vulkan renderers. DirectX 11 is a trusted renderer for most PC gamers, giving optimal performance with few issues. Vulkan may offer better performance for some users, depending on the system.
*   **FidelityFX Super Resolution (FSR):** FSR is an upscaling technology that can improve performance with minimal visual quality loss.

### Current Status and Developments

As of early 2026, optimization guides and settings are continuously being updated by the community. Pro players often share their configurations, and community members conduct tests to determine the impact of various settings on FPS. The game is still relatively new, so optimization techniques are likely to evolve.

### Practical Implications

The practical implications of optimizing "Deadlock" are significant for competitive players. Higher FPS and lower input latency can lead to improved reaction times and aiming accuracy. Stable performance is also crucial for consistent gameplay and avoiding frustrating stutters or freezes. By following optimization guides and tailoring settings to their specific hardware, players can gain a competitive edge.

### Important Caveats or Limitations

*   **Hardware Dependence:** The "best" settings will vary depending on individual hardware configurations. What works well for a high-end PC may not be suitable for a low-end system.
*   **Subjectivity:** Some settings, such as crosshair preferences and visual quality, are subjective and depend on personal taste.
*   **Game Updates:** Game updates can change the impact of certain settings, so it's important to stay up-to-date with the latest optimization guides.
*   **Early Access:** As "Deadlock" may still be in development, performance issues and optimization opportunities may change as the game is further refined.

### Key Sources and References

1.  \*UPDATED\* Best COMPETITIVE settings for DEADLOCK V2 - YouTube ([https://www.youtube.com/watch?v=azlmN4QHWBs](https://www.youtube.com/watch?v=azlmN4QHWBs))
2.  For the surprise of 0 people, Deadlock, the new Valve game works without any issues on Linux : r/linux\_gaming - Reddit ([https://www.reddit.com/r/linux\_gaming/comments/162x9bu/for\_the\_surprise\_of\_0\_people\_deadlock\_the\_new/](https://www.reddit.com/r/linux_gaming/comments/162x9bu/for_the_surprise_of_0_people_deadlock_the_new/))
3.  CS2 Best Settings & Options Guide - ProSettings.net ([https://prosettings.net/cs-settings-options-guide/](https://prosettings.net/cs-settings-options-guide/))
4.  \*UPDATED\* Best COMPETITIVE settings for DEADLOCK V5 - YouTube ([https://www.youtube.com/watch?v=f-DT_-KhnjQ](https://www.youtube.com/watch?v=f-DT_-KhnjQ))
5.  Best Deadlock Settings & Options Guide For Max FPS - Eloboss ([https://eloboss.com/deadlock-best-settings/](https://eloboss.com/deadlock-best-settings/))
6.  \[NEW] BEST Optimization Guide | Deadlock | Max FPS | Best Settings - YouTube ([https://www.youtube.com/watch?v=fM1I9nTTCKk](https://www.youtube.com/watch?v=fM1I9nTTCKk))
7.  Best Deadlock Settings: for FPS, Graphics, Audio, Mouse - Skycoach ([https://skycoach.gg/blog/deadlock/best-deadlock-settings-2026](https://skycoach.gg/blog/deadlock/best-deadlock-settings-2026))
8.  Command line options - Valve Developer Community ([https://developer.valvesoftware.com/wiki/Command\_Line\_Options](https://developer.valvesoftware.com/wiki/Command_Line_Options))
9.  Best Deadlock Settings to Boost FPS and Performance - Hawk Live ([https://hawklive.gg/settings/best-deadlock-settings-to-boost-fps-and-performance/](https://hawklive.gg/settings/best-deadlock-settings-to-boost-fps-and-performance/))
10. CS2 Config Optimization Guide (2026): Boost FPS & Fix Lag ([https://www.youtube.com/watch?v=LNhypP3yZfQ](https://www.youtube.com/watch?v=LNhypP3yZfQ))
11. BEST DEADLOCK SETTINGS + LAUNCH OPTIONS - YouTube ([https://www.youtube.com/watch?v=wY9eWtq-JR4](https://www.youtube.com/watch?v=wY9eWtq-JR4))
12. Deadlock | Every Setting Tested | Graphics Breakdown & Performance - YouTube ([https://www.youtube.com/watch?v=F4KC1AAFEUI](https://www.youtube.com/watch?v=F4KC1AAFEUI))
13. ConVars - Introduction | CS2 Docs ([https://developer.valvesoftware.com/wiki/ConVars](https://developer.valvesoftware.com/wiki/ConVars))
14. \ Deadlock - Boost your FPS with these essential PC settings! - YouTube ([https://www.youtube.com/watch?v=Cix6IgCVjVY](https://www.youtube.com/watch?v=Cix6IgCVjVY))
15. Setting Game Launch Options - Steam Support ([https://help.steampowered.com/en/faq/view/7D01-D2DD-D75E-2955](https://help.steampowered.com/en/faq/view/7D01-D2DD-D75E-2955))
16. The ULTIMATE PC Optimization Guide for 2026 - YouTube ([https://www.youtube.com/watch?v=n9jDniNUnOE](https://www.youtube.com/watch?v=n9jDniNUnOE))
17. cvarlist - Valve Developer Community ([https://developer.valvesoftware.com/wiki/cvarlist](https://developer.valvesoftware.com/wiki/cvarlist))
18. Is this game made in Source or Source 2? : r/DeadlockTheGame - Reddit ([https://www.reddit.com/r/DeadlockTheGame/comments/wkvq46/is\_this\_game\_made\_in\_source\_or\_source\_2/](https://www.reddit.com/r/DeadlockTheGame/comments/wkvq46/is_this_game_made_in_source_or_source_2/))
19. Guide :: Launch Options - Counter-Strike - Steam Community ([https://steamcommunity.com/sharedfiles/filedetails/?id=807737818](https://steamcommunity.com/sharedfiles/filedetails/?id=807737818))
20. CS2 Best Settings & FPS Boost 2026 | Complete Optimization Guide - YouTube ([https://www.youtube.com/watch?v=nuZnH3JtW3w](https://www.youtube.com/watch?v=nuZnH3JtW3w))
21. How is deadlock performing on your pc? : r/DeadlockTheGame - Reddit ([https://www.reddit.com/r/DeadlockTheGame/comments/10583ni/how\_is\_deadlock\_performing\_on\_your\_pc/](https://www.reddit.com/r/DeadlockTheGame/comments/10583ni/how_is_deadlock_performing_on_your_pc/))
22. Source 2 - Valve Developer Community ([https://developer.valvesoftware.com/wiki/Source\_2](https://developer.valvesoftware.com/wiki/Source_2))
23. Command line options (Source 2) - Valve Developer Community ([https://developer.valvesoftware.com/wiki/Command\_line\_options\_(Source\_2)](https://developer.valvesoftware.com/wiki/Command_line_options_(Source_2)))
24. Boost Your FPS and Spot Enemies Easily in Deadlock - YouTube ([https://www.youtube.com/watch?v=3exjfFjSJyY](https://www.youtube.com/watch?v=3exjfFjSJyY))
25. Half-Life 2 / Counter-Strike Source Cvar List - basicer ([https://basicer.de/half-life-2-counter-strike-source-cvar-list/](https://basicer.de/half-life-2-counter-strike-source-cvar-list/))
26. Defining custom launch options for a Sourcemod without user input? - Reddit ([https://www.reddit.com/r/sourcemods/comments/1003596/defining\_custom\_launch\_options\_for\_a\_sourcemod/](https://www.reddit.com/r/sourcemods/comments/1003596/defining_custom_launch_options_for_a_sourcemod/))
27. How to Optimize CS2 in 2026? | A Complete Guide to 300+ FPS (New Methods) - YouTube ([https://www.youtube.com/watch?v=PqJSA\_hm0xI](https://www.youtube.com/watch?v=PqJSA_hm0xI))
28. CSGO and CS2 Convars Overview | PDF | Server (Computing) - Scribd ([https://www.scribd.com/document/687082244/CSGO-and-CS2-Convars-Overview](https://www.scribd.com/document/687082244/CSGO-and-CS2-Convars-Overview))
29. How to export the full cvar list of any Source Engine game - Steam Community ([https://steamcommunity.com/sharedfiles/filedetails/?id=3072694424](https://steamcommunity.com/sharedfiles/filedetails/?id=3072694424))
30. Learning Deadlock Day 2 (Emissary 1) - YouTube ([https://www.youtube.com/watch?v=y\_j6OGEWz3k](https://www.youtube.com/watch?v=y_j6OGEWz3k))
31. How Deadlock RUINED its Visual Clarity - YouTube ([https://www.youtube.com/watch?v=QfsQj-ILWBI](https://www.youtube.com/watch?v=QfsQj-ILWBI))


## Sources
- [valvesoftware.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEcmEv7VUqvO8htJV6Neuh4ZiAfN3--39R_Y5-M_acklpM_ZX2upYmdOAZcToQYjWALgwCrzfDgVX05fMWPWxq87MWb8U8ExWADD_J3G3K1iPcMRsIuTXa49Eq9A0VpxAqu6w-EnSDQIS9SVfw=)
- [skycoach.gg](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGnyzLb7b5gr3b2D_ZOcleCh76hHLUErvREv1VvzLD7t3hy9YbZk25T2DZQLjo3pTCMih3Tw3aYqa-MBAL_LH-2wIRtu6wQAJNuXA4sJftRcLbNAYsMWaR3Z2kbn5NbXQqaEa6IBjFTpPEcoXjVSdG7r-wqT2OC3dvGeTVg)
- [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH_pciZwBtOUg6oVKpx7xQeDKRYCx76lhYpq-_FGCuwaKV6keh_ViPecCU7EKrDOIyH-hByxrEzUZ1FTqXRiSEb8ITywvj7P7Z2Ii8Nn5WUWFqZjnvYKBViMdqWv_Rm_iIN0cYid9E=)
- [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE0DP6SeGoZkHiTUIJo3ZQ7IXQoP9AYVb0HuXdY--oOvHywh8r-KNOSTyxILTU2xM8-2d2s8ws36WUg22Za2xaiQao_kb71poIHuZw7gLdjTxfayX6n4xS1oYTnDkEepxPnki8dOCA=)
- [eloboss.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQED_OZHX0sb_69YJDXNmBTNTmHBkwfJDgvukwqcOM4BVXvhUYAj-SzzeIvJ5gyxrB_c4Ogv5Uik-PL0IG1ddT_tImCJp5hb5coCMKnkCirJMtEjjn5UVKXMe11cbabe-MDnVlwXLPJ8H3bu)
- [valvesoftware.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEeej-liJ9m9AnHb5auDyaoU8tTa5tREO9usCfpV9ThSSnyR5Yotbe4owB_oVDNOlDGXMFF8myM8o4R4Qy_5RK4LqHrKCdilCzr1Vb2EmCWYWKlGOqMov0QyGiBqGh1l4LB2QRs-umbaid5fhFsnxnw6DmtOeTXPtM=)
- [valvesoftware.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE_qbQob3cqJ5VweVaunstZFSC6R36zvusAIsaoOeQzGn4HWfnl9Rab5gZIWEBBURg62dJciia2Gs7sBlGOvV0GsSsZ7eaUN3dZRN62awcHLSvtlqtCAzthfDnusSXokhME2zjgm5pf_rs1KhJ-23Wj70VsWLN1vnxiYXgEHhUaMOQWlw==)
- [steampowered.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF0-CCtHF8SbQ6wL1VDP_mpD-Tbi8NRV5uC5hYGaYl0SNtKm0XSwerWZikcFvyLbasf7vOACH_au0jNnDZUVyEgUNGPJNOg_NkVEAMWH0zz_X8LJ8okE3uK4VDjxMywYcz1-TiMvA_4fLJXPVe03-R4PIbkfH8-HPs3)
- [fpscalculator.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGKmrDVq5Gv41hD3iLymwYfx_hDtsSvaBUR47jQLDQrd8ksvs2YotSIxieVzTLUuXhRylBAqzJ5GaQc0dPx-uUsbdA-yZEkFgHJcOXmcTCwyeOZLZjLjMOc9l_RAhJIrlCTH8h6YiXNxh_XUQVM-YdM4yDK-m_hlkXgCKwBjyAk5fJJZ1PCLKdWPLfBsbfoPpQ748Q=)
- [prosettings.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHQd4srMExj0YHlFzDTXUQvEg-fwA5GVh1QtfawXc96IgKi1QqbMSIVCTKO1Z7ZcLJNv_8rcpUKLk6PeAoBaBHXrMbi-660nMrOJpSEYKvwfZuCtOTNEEmqTWdC9JV6L7e-0I3lXzA=)
- [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG54b0GHzxpYw7XA9R2wDfR5Owj-GSTLumypGySm-aKFCKWWR0HCjQEBleTICtofsl3YU--pjK-ePlnJkal9gghEzibjvkB4OozOOc2doy9A238xFxp_4HpmsI0JfSrlY-ycBPqeNs=)
- [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFMrIXpu0HIG-hFIpwnAitZKWHn9Bd_tDumB2A3Og_0sFzVmbB_4i7aUlRaJBp3_pXc05C80Jhrlz3WQnKXJgmZ_CmiWfw5E16i6ZxN1EhScR8ElBArASGuylDyWPem_ETaHX59S-E=)
- [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHRp9y6p7KyQdT9wDyLuLXsDZDR7d0SZbfs8Qcwg4oZ9nhLCPz7-TlSBDvtGrgSjFXsBASVOtSNLpqPPtAqMLtT7_qmQrwEt1-8ap8A_whvKw3u2J9L8h_PsDofKWn1jGbJE5mYCDc=)
- [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGLRw_EWQnZyBG-qKrgEBKXdP-1yJ2WskOHURnS57vyGjtFyjsK6IHiMVb3mhlx_0qKsQGRHiGZ4eV-MFlOm96zstPuDlVzlQ4JuHm5ZHFaRPcn4Ev8cE5eu_zYb1l7dD3_7WtnLPY=)
- [valvesoftware.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEHVjledxC-dBCyvNBrq9lVnwHRXan5Vf1pxQB0fug4h8zVJtY4rsenkCrLjuf2wfmqiS2l0nQnmerNbL27ML1gtQCrJ853pfABnR3keHoG6H1xi_t6dqkEyNbbUkhGkBfK1HkOCh_FW1YOwcI=)
- [basicer.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsxP73V5H-Vgzv_DfUmFCICyHpjXq5YKvmWh-eQkz2lqVeta3X8kN3pEpBf1BVj2zhIp2-twqdO9YkxbmTn04AtnVjHlE7cJngdT_1JJB64vR5Ip_5WUy2FcI6iJ0BNvIDocI=)
- [poggu.me](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEqndlEj5WCg3rA4ARNdyeJ9O_h8EbUSQocu1eR2oDzpDZs2gntbKMyOd3jTq30tGmJmZ05TtxD86YqtyyHIYIB5heZZ3oWKlggibtUIAc5-VDJTY8WDRQ-4k2_G8DJlzV8HOsH4X2GJQ==)
- [steamcommunity.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEr9IkKaxUloNy7DulGq2IKn4PDBgJlVQOVgEqSL4ifBVKgf0_yanAmsMk1rI4-Fce1bcjRNMIwfAlh0v4MuFLkQ7PBYNREW8W3RVfIfxwnaQ4ej6UmshwNfNOA-K_MkMl9Fii6W_dUvUgP4Nlaic4uV4KmsplpFqcjw3RX)
